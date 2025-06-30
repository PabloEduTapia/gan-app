import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2DTranspose, Conv2D, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Cargar dataset
def load_images_from_folder(folder, image_size=(64, 64)):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = load_img(os.path.join(folder, filename), target_size=image_size)
            images.append(img_to_array(img))
    images = np.array(images)
    return (images - 127.5) / 127.5  # Normalización

dataset_path = 'dataset/selfishgene/synthetic-faces-high-quality-sfhq-part-1/versions/4/images/images'
X_train = load_images_from_folder(dataset_path)
print(f"✅ {X_train.shape[0]} imágenes cargadas")

latent_dim = 100
img_shape = (64, 64, 3)

def build_generator():
    model = Sequential()
    model.add(Dense(256 * 8 * 8, input_dim=latent_dim))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((8, 8, 256)))
    model.add(Conv2DTranspose(128, 4, 2, 'same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(64, 4, 2, 'same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(3, 4, 2, 'same', activation='tanh'))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, 4, 2, 'same', input_shape=img_shape))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, 4, 2, 'same'))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

generator = build_generator()
discriminator.trainable = False

gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

# Crear carpeta para modelos
os.makedirs("modelos", exist_ok=True)
os.makedirs("muestras", exist_ok=True)

def save_generated_image(epoch):
    noise = np.random.normal(0, 1, (1, latent_dim))
    gen_img = generator.predict(noise)[0]
    gen_img = ((gen_img * 127.5) + 127.5).astype(np.uint8)
    img = Image.fromarray(gen_img)
    img.save(f"muestras/sample_epoch_{epoch}.png")

def train(epochs=10000, batch_size=64, save_interval=500):
    for epoch in range(1, epochs + 1):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 100 == 0:
            print(f"🧪 Epoch {epoch} - D real: {d_loss_real[0]:.4f}, D fake: {d_loss_fake[0]:.4f}, G: {g_loss:.4f}")

        if epoch % save_interval == 0:
            generator.save(f"modelos/modelo_gan_{epoch}.h5")
            save_generated_image(epoch)
            print(f"💾 Modelo y muestra guardados en epoch {epoch}")

train()
