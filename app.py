import os
import tempfile
import logging
from flask import Flask, request, jsonify
from deepface import DeepFace
import traceback
import cv2

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_NAME = "Facenet"
DISTANCE_THRESHOLD = 0.68

@app.route('/verify', methods=['POST'])
def verify():
    try:
        if 'selfie' not in request.files or 'reference' not in request.files:
            return jsonify({"error": "Both 'selfie' and 'reference' files are required."}), 400

        selfie_file = request.files['selfie']
        ref_file = request.files['reference']

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_selfie:
            selfie_file.save(tmp_selfie.name)
            selfie_path = tmp_selfie.name

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_ref:
            ref_file.save(tmp_ref.name)
            ref_path = tmp_ref.name

        try:
            selfie_faces = DeepFace.extract_faces(img_path=selfie_path, detector_backend='opencv', enforce_detection=False)
            ref_faces = DeepFace.extract_faces(img_path=ref_path, detector_backend='opencv', enforce_detection=False)
        except Exception as e:
            os.unlink(selfie_path)
            os.unlink(ref_path)
            return jsonify({"error": f"Face detection failed: {str(e)}"}), 400

        num_selfie_faces = len(selfie_faces)
        num_ref_faces = len(ref_faces)

        if num_selfie_faces == 0:
            os.unlink(selfie_path)
            os.unlink(ref_path)
            return jsonify({
                "verified": False,
                "status": "No Face",
                "distance": None,
                "num_faces_selfie": 0,
                "num_faces_reference": num_ref_faces
            })

        if num_ref_faces != 1:
            os.unlink(selfie_path)
            os.unlink(ref_path)
            return jsonify({"error": f"Reference image contains {num_ref_faces} faces, but exactly 1 is required."}), 400

        best_distance = float('inf')
        any_match = False

        try:
            ref_face_img = ref_faces[0]['face']
            ref_face_path = tempfile.mktemp(suffix='.jpg')
            cv2.imwrite(ref_face_path, ref_face_img)
        except Exception as e:
            os.unlink(selfie_path)
            os.unlink(ref_path)
            return jsonify({"error": f"Failed to process reference face: {str(e)}"}), 500

        for i, face_data in enumerate(selfie_faces):
            try:
                face_img = face_data['face']
                face_path = tempfile.mktemp(suffix='.jpg')
                cv2.imwrite(face_path, face_img)

                result = DeepFace.verify(img1_path=face_path, img2_path=ref_face_path,
                                         model_name=MODEL_NAME, distance_metric='cosine',
                                         enforce_detection=False, detector_backend='skip')
                distance = result['distance']
                if distance < best_distance:
                    best_distance = distance
                if result['verified']:
                    any_match = True
                os.unlink(face_path)
            except Exception as e:
                logging.warning(f"Face {i} verification error: {e}")

        os.unlink(selfie_path)
        os.unlink(ref_path)
        os.unlink(ref_face_path)

        if any_match:
            status = "Pass"
            verified = True
        else:
            if num_selfie_faces > 1:
                status = "Multiple faces, no match"
            else:
                status = "Failed"
            verified = False

        return jsonify({
            "verified": verified,
            "status": status,
            "distance": round(best_distance, 4) if best_distance != float('inf') else None,
            "num_faces_selfie": num_selfie_faces,
            "num_faces_reference": num_ref_faces
        })

    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
