from src.utils.id_utils import FaceAnalysis2, analyze_faces
import numpy as np


class Aligner():
    def __init__(self):
        self.face_detector = FaceAnalysis2(root="/home/jovyan/bobkov/insightface_models", providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))

    def __call__(self, input_images):
        bboxes = []
        embeds = []
        for img in input_images:
            max_img_size = max(img.size)
            img = np.array(img)
            img = img[:, :, ::-1]
            faces = analyze_faces(self.face_detector, img)
            if len(faces) == 0:
                bboxes.append(None)
                embeds.append(None)
                continue

            face_embeds = []
            face_bboxes = []
            for face in faces:
                bbox = face['bbox'].astype(np.int32)
                bbox = np.clip(bbox, 0, max_img_size)
                face_bboxes.append(bbox.tolist())
                face_embeds.append(face['embedding'])
                
            embeds.append(face_embeds)
            bboxes.append(face_bboxes)
        
        return bboxes, embeds