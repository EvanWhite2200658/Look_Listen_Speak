from __future__ import annotations

import json
import pathlib

import cv2 as cv
import numpy as np
from gazefollower.calibration.SVRCalibration import SVRCalibration
from gazefollower.logger import Log


class PatchedSVRCalibration(SVRCalibration):
    """
    Project-local patch for GazeFollower SVR calibration.

    Fixes calibration training when features arrive as a 3D tensor:
        (num_targets, samples_per_target, num_features)

    by reshaping to:
        (num_targets * samples_per_target, num_features)

    and repeating labels accordingly.

    Also stores the trained feature count so runtime prediction can
    validate input shape before calling OpenCV.
    """

    def __init__(self, model_save_path: str = ""):
        # do not rely on parent auto-load behavior
        super().__init__(model_save_path)
        if model_save_path == "":
            self.workplace_calibration_dir = pathlib.Path.home().joinpath("GazeFollower", "calibration")
            if not self.workplace_calibration_dir.exists():
                self.workplace_calibration_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.workplace_calibration_dir = pathlib.Path(model_save_path)
            self.workplace_calibration_dir.mkdir(parents=True, exist_ok=True)

        self.svr_x_path = self.workplace_calibration_dir.joinpath("svr_x.xml")
        self.svr_y_path = self.workplace_calibration_dir.joinpath("svr_y.xml")
        self.metadata_path = self.workplace_calibration_dir.joinpath("svr_metadata.json")

        self.expected_feature_count: int | None = None

        if self.metadata_path.exists():
            try:
                data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                self.expected_feature_count = int(data["feature_count"])
                print(f"[PatchedSVRCalibration] loaded expected_feature_count={self.expected_feature_count}")
            except Exception as exc:
                print(f"[PatchedSVRCalibration] metadata load warning: {exc}")

        self.has_calibrated = False

        if self.svr_x_path.exists() and self.svr_y_path.exists():
            try:
                self.svr_x = cv.ml.SVM_load(str(self.svr_x_path))
                self.svr_y = cv.ml.SVM_load(str(self.svr_y_path))
                self.has_calibrated = True
                print("[PatchedSVRCalibration] loaded saved SVM models with cv.ml.SVM_load")
            except Exception as exc:
                print(f"[PatchedSVRCalibration] model load warning: {exc}")
                self.svr_x = cv.ml.SVM.create()
                self.svr_y = cv.ml.SVM.create()
                self._set_svm_params(self.svr_x)
                self._set_svm_params(self.svr_y)
                self.has_calibrated = False
        else:
            self.svr_x = cv.ml.SVM.create()
            self.svr_y = cv.ml.SVM.create()
            self._set_svm_params(self.svr_x)
            self._set_svm_params(self.svr_y)

    def _save_metadata(self) -> None:
        if self.expected_feature_count is None:
            return
        payload = {"feature_count": self.expected_feature_count}
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def calibrate(self, features, labels, ids=None):
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)

        print(f"[PatchedSVRCalibration] incoming features shape: {features.shape}")
        print(f"[PatchedSVRCalibration] incoming labels shape: {labels.shape}")

        if features.ndim == 3:
            num_targets, samples_per_target, num_features = features.shape
            features = features.reshape(num_targets * samples_per_target, num_features)

            if labels.ndim == 2 and labels.shape[0] == num_targets:
                labels = np.repeat(labels, samples_per_target, axis=0)
            else:
                raise ValueError(
                    "Label shape is incompatible with 3D feature tensor. "
                    f"Got labels.shape={labels.shape}, expected first dim {num_targets}."
                )

        elif features.ndim != 2:
            raise ValueError(
                "Expected features to be 2D or 3D for calibration. "
                f"Got shape {features.shape}."
            )

        self.expected_feature_count = int(features.shape[1])

        print(f"[PatchedSVRCalibration] training features shape: {features.shape}")
        print(f"[PatchedSVRCalibration] training labels shape: {labels.shape}")
        print(f"[PatchedSVRCalibration] expected_feature_count={self.expected_feature_count}")

        labels_x = labels[:, 0].reshape(-1, 1)
        labels_y = labels[:, 1].reshape(-1, 1)

        try:
            self.svr_x.train(features, cv.ml.ROW_SAMPLE, labels_x)
            self.svr_y.train(features, cv.ml.ROW_SAMPLE, labels_y)
            self.has_calibrated = True
        except Exception as e:
            self.has_calibrated = False

            Log.e(f"Failed to train patched SVM model: {e.args}")
            Log.d("Try to delete previously trained model.")

            if self.svr_x_path.exists():
                self.svr_x_path.unlink()
                Log.d(f"Deleted: {self.svr_x_path}")

            if self.svr_y_path.exists():
                self.svr_y_path.unlink()
                Log.d(f"Deleted: {self.svr_y_path}")

            if self.metadata_path.exists():
                self.metadata_path.unlink()

        if self.has_calibrated:
            predicted_x = self.svr_x.predict(features)[1]
            predicted_y = self.svr_y.predict(features)[1]

            euclidean_distances = np.sqrt((labels_x - predicted_x) ** 2 + (labels_y - predicted_y) ** 2)
            mean_euclidean_error = np.mean(euclidean_distances)

            Log.d(f"Patched calibration completed with mean Euclidean error: {mean_euclidean_error:.4f}")

            predictions = np.concatenate((predicted_x, predicted_y), axis=1)
            return self.has_calibrated, mean_euclidean_error, predictions

        return self.has_calibrated, float("inf"), None

    def save_model(self) -> bool:
        saved = super().save_model()
        if saved:
            try:
                self._save_metadata()
                print(f"[PatchedSVRCalibration] saved metadata at {self.metadata_path}")
            except Exception as exc:
                print(f"[PatchedSVRCalibration] metadata save warning: {exc}")
        return saved

    def predict(self, features, estimated_coordinate):
        features = np.asarray(features, dtype=np.float32)
        print(f"[PatchedSVRCalibration] predict incoming shape: {features.shape}")

        if features.ndim == 1:
            actual_feature_count = int(features.shape[0])
            features_2d = features.reshape(1, -1)
        elif features.ndim == 2 and features.shape[0] == 1:
            actual_feature_count = int(features.shape[1])
            features_2d = features
        else:
            print(
                "[PatchedSVRCalibration] unsupported runtime feature shape for prediction: "
                f"{features.shape}. Returning estimated coordinate."
            )
            return False, estimated_coordinate

        print(
            f"[PatchedSVRCalibration] predict feature count={actual_feature_count}, "
            f"expected={self.expected_feature_count}"
        )

        if not self.has_calibrated:
            print("[PatchedSVRCalibration] no calibrated model available. Returning estimated coordinate.")
            return False, estimated_coordinate

        if self.expected_feature_count is not None and actual_feature_count != self.expected_feature_count:
            print(
                "[PatchedSVRCalibration] feature count mismatch. "
                f"expected {self.expected_feature_count}, got {actual_feature_count}. "
                "Returning estimated coordinate instead of crashing."
            )
            return False, estimated_coordinate

        try:
            predicted_x = self.svr_x.predict(features_2d)[1].flatten()[0]
            predicted_y = self.svr_y.predict(features_2d)[1].flatten()[0]
            return True, (predicted_x, predicted_y)
        except Exception as exc:
            print(f"[PatchedSVRCalibration] predict warning: {exc}")
            return False, estimated_coordinate