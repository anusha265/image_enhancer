from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from skimage import exposure, transform
import cv2

def home(request):
    return render(request, 'enhancer/home.html')

def process_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']

        # Save the uploaded image to a temporary location
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)

        # Read the image using OpenCV
        img = cv2.imread(fs.path(filename))

        # Perform image enhancement using different techniques
        enhanced_img = enhance_image(img)

        # Increase image resolution
        increased_resolution_img = increase_resolution(enhanced_img)

        # Save the processed image to a new location
        processed_filename = 'processed_' + filename
        cv2.imwrite(fs.path(processed_filename), increased_resolution_img)

        # Remove the temporary uploaded image
        fs.delete(filename)

        # Redirect to download the processed image
        return redirect('download', processed_filename=processed_filename)

    return redirect('home')

def download(request, processed_filename):
    fs = FileSystemStorage()
    processed_file_path = fs.path(processed_filename)
    with open(processed_file_path, 'rb') as file:
        response = HttpResponse(file.read(), content_type='image/jpeg')
        response['Content-Disposition'] = 'attachment; filename=' + processed_filename
        return response

def enhance_image(img):
    # Apply advanced image enhancement techniques
    enhanced_img = exposure.adjust_gamma(img, gamma=0.8)
    enhanced_img = exposure.rescale_intensity(enhanced_img, in_range='image', out_range='dtype')
    enhanced_img = exposure.adjust_sigmoid(enhanced_img, cutoff=0.5, gain=10)

    return enhanced_img

def increase_resolution(img):
    # Increase image resolution
    increased_resolution_img = transform.resize(img, (img.shape[0] * 2, img.shape[1] * 2), preserve_range=True)

    return increased_resolution_img
