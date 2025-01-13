from ai_modules import llama
from ai_modules import real_esrgan_and_clahe
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

LLaMA = llama.Llama
Vismodel = real_esrgan_and_clahe.VisModel
# Getting models from ai_modules and plotting the images
def process_image(image):
    model = Vismodel(image)
    gan = model.pred_gan(image)
    clahe = model.pred_gan_with_clahe(image)
    # Matplotlib ile görselleri yan yana çiz
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Orijinal Resim")
    axes[0].axis("off")

    axes[1].imshow(gan)
    axes[1].set_title("Real-ESRGAN")
    axes[1].axis("off")

    axes[2].imshow(clahe)
    axes[2].set_title("Real-ESRGAN + CLAHE")
    axes[2].axis("off")

    # Grafiği bir görsele dönüştür
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    combined_image = Image.open(buf)
    
    return combined_image

# Product description function using LLaMA
def product_description(description):
    llama_model = LLaMA(description)
    response = llama_model.generate_response()
    return response


# Gradio interface
with gr.Blocks() as demo:
    # Creating tabs
    with gr.Tab("Ürün Görsel Kalitesi Arttırma"):
        with gr.Row():
            input_image = gr.Image(label="Upload Image", type="numpy")
            submit_button = gr.Button("Resmi İşle")
            
        with gr.Row():
            output_plot = gr.Image(label="İşlenmiş Resimler")
            
        submit_button.click(
            process_image,
            inputs=[input_image],
            outputs=[output_plot],  # Tek bir matplotlib çıktısı
        )

    with gr.Tab("Ürün Açıklaması"):
        description_input = gr.Textbox(label="Ürün Açıklaması Girin")
        description_output = gr.Textbox(label="Çıktı")
        
        # Ürün açıklaması fonksiyonu
        description_input.submit(product_description, inputs=description_input, outputs=description_output)

# Starting app
demo.launch()

