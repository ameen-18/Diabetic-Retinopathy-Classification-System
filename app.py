import io
import os
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# Load the VGG16 model
vgg16_model = load_model('vgg16_model.h5')

# Define the path to the ResNet model file
resnet_model_path = os.path.join(os.path.dirname(__file__), 'resnet18_model.pth')

# Load the ResNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
resnet_model = models.resnet18(pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device))
resnet_model.eval()

# Define image transformations for ResNet model
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict using VGG16 model
def predict_vgg16(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    predicted_classes = vgg16_model.predict(img_array)
    predicted_class_index = np.argmax(predicted_classes)
    class_labels = ['No', 'Mild', 'Moderate', 'Severe', 'Proliferate']
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

# Function to predict using ResNet model
def predict_resnet(image_path):
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = resnet_model(img_tensor)
    predicted_class_index = torch.argmax(outputs).item()
    class_labels = ['No', 'Mild', 'Moderate', 'Severe', 'Proliferate']
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

dict={'No':'''There is nothing to worry .
      Everything seems perfect ''',
      'Mild':'''Continue regular eye exams: Increase the frequency of eye exams to monitor progression and assess the need for treatment.          
      Optimize diabetes management: Work closely with healthcare providers to control blood sugar, blood pressure, and cholesterol levels.
      Consider additional interventions: Depending on individual risk factors, your eye care professional may recommend additional tests or interventions to monitor and manage the condition.''',
      'Moderate':'''Continue regular eye exams: Increase the frequency of eye exams to monitor progression and assess the need for treatment.
      Optimize diabetes management: Work closely with healthcare providers to control blood sugar, blood pressure, and cholesterol levels.
      Consider additional interventions: Depending on individual risk factors, your eye care professional may recommend additional tests or interventions to monitor and manage the condition''',
      'Severe':'''Follow treatment recommendations: If signs of severe NPDR are detected, your eye care professional may recommend treatment options such as intravitreal injections or laser therapy to prevent progression to PDR.
      Manage systemic health: Address underlying health conditions such as hypertension and hyperlipidemia to reduce the risk of worsening retinopathy.''',
      'Proliferate':'''Undergo prompt treatment: Seek treatment promptly if diagnosed with PDR, as it carries a higher risk of vision loss.Consider laser surgery: Laser photocoagulation or photocoagulation therapy may be recommended to seal leaking blood vessels and prevent further proliferation.
      Monitor for complications: Regular follow-up visits with an eye care professional are essential to monitor for complications such as vitreous hemorrhage or retinal detachment.'''}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected")

        if file:
            img_path = "uploads/image.png"
            file.save(img_path)

            # Predict using VGG16 model
            vgg16_prediction = predict_vgg16(image.load_img(img_path, target_size=(128, 128)))

            # Predict using ResNet model
            resnet_prediction = predict_resnet(img_path)

            os.remove(img_path)


            

            return render_template('inner-page.html', vgg16_result=vgg16_prediction,vgresult_desc=dict[vgg16_prediction])

if __name__ == '__main__':
    app.run(debug=True)
