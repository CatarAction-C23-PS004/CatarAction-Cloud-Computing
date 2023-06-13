from flask import Flask, request, jsonify
import numpy as np
import os
import uuid
import tensorflow as tf
import tensorflow_hub as hub
from firebase_admin import credentials, db, initialize_app

tf.keras.utils.get_custom_objects()['KerasLayer'] = hub.KerasLayer

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'

app.config["DEBUG"] = True

cred = credentials.Certificate('key.json')
initialize_app(cred, {'databaseURL': 'https://database-api-30a74-default-rtdb.firebaseio.com/history'})

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    image_name = image.filename

    if image_name == '':
        return jsonify({'error': 'Image name not provided'}), 400

    try:
        image_id = str(uuid.uuid4())
        image_path = os.path.join(UPLOAD_FOLDER, image_name)
        image.save(image_path)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32') / 255.0
        model = tf.keras.models.load_model('model.h5')
        output = model.predict(image)
        response = {'class': output.tolist()}
        predicted_class = 'Cataract' if response['class'][0][0] > response['class'][0][1] else 'Normal'

        data = {
            'image_id': image_id,
            'image_name': image_name,
            'result': predicted_class
        }
        ref = db.reference('/history')
        new_data_ref = ref.push()
        new_data_ref.set(data)

        return jsonify({'class': predicted_class}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_data():
    ref = db.reference('/history')
    data = ref.get()

    return jsonify(data)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run()