from flask import Flask, render_template, request, redirect, url_for
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

app = Flask(__name__)

model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    uploaded_file = request.files['image']
    if uploaded_file.filename != '':
        uploaded_file.save('uploaded_image.jpg')
        image = Image.open('uploaded_image.jpg')

        prompt = "<grounding>An image of"  # You can customize the prompt here

        inputs = processor(text=prompt, images=image, return_tensors="pt")
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        processed_text, entities = processor.post_process_generation(generated_text)

        return render_template('output.html', generated_text=processed_text, entities=entities)
    else:
        return redirect(url_for('index'))  # Redirect to the input page if no image is uploaded

if __name__ == '__main__':
    app.run(debug=True)
