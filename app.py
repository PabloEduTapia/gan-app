from flask import Flask, render_template, send_file
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Cargar modelo entrenado desde Colab
generator = load_model("modelo_gan_colab.keras")
latent_dim = 100  # igual al usado en Colab

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    # Generar 5 imágenes
    noise = np.random.normal(0, 1, (5, latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # reescalar de [-1, 1] a [0, 1]

    # Crear figura con matplotlib
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(gen_imgs[i])
        axs[i].axis('off')
    plt.tight_layout()

    # Convertir a imagen en memoria
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
