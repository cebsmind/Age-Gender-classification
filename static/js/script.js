// script.js

// Function to handle file selection and display image preview
function previewImage() {
    var fileInput = document.getElementById('fileInput');
    var imagePreview = document.getElementById('imagePreview');

    // Check if a file is selected
    if (fileInput.files && fileInput.files[0]) {
        var reader = new FileReader();

        // Set up the reader to read the selected file
        reader.onload = function (e) {
            // Display the image preview
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        };

        // Read the selected file as a data URL
        reader.readAsDataURL(fileInput.files[0]);
    }
}

// script.js

// Function to validate the form before submission
function validateForm() {
    var fileInput = document.getElementById('fileInput');

    // Check if a file is selected
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select an image before submitting.');
        return false; // Prevent form submission
    }

    return true; // Allow form submission
}

// type effect .js
function typeEffect(element, text, speed) {
    let i = 0;

    function type() {
        if (i < text.length) {
            element.innerHTML += "<strong>" + text.charAt(i) + "</strong>";
            i++;
            setTimeout(type, speed);
        }
    }

    type();
}

// Call the typing effect for the heading and paragraph
const headingElement = document.getElementById('typed-heading');
const textElement = document.getElementById('typed-text');

// typeEffect(headingElement, 'Welcome to my page :-)');
typeEffect(textElement, 'Hello ! I\'m C.A.I, an AI bot designed by my master : Cebrail. He\'s just a random guy passionate about AI and stuff. My goal is simple : Give me a picture and I will try to guess your age and gender. Have fun playing around!', 30);