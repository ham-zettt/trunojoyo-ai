# Trunojoyo AI

Trunojoyo AI is a FastAPI-based web application that utilizes Generative Adversarial Networks (GAN) for generating batik designs, processing facial images, and recognizing sign language. This project aims to explore the intersection of traditional art and modern technology.

## Project Structure

```
trunojoyo-ai
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── api
│   │   ├── __init__.py
│   │   └── routes
│   │       ├── __init__.py
│   │       ├── batik.py
│   │       ├── face.py
│   │       └── sign_language.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── gan_model.py
│   │   └── model_utils.py
│   ├── services
│   │   ├── __init__.py
│   │   ├── batik_generator.py
│   │   ├── face_processor.py
│   │   └── sign_language_processor.py
│   └── static
│       ├── css
│       │   └── style.css
│       ├── js
│       │   └── app.js
│       └── assets
├── templates
│   ├── base.html
│   ├── index.html
│   ├── batik.html
│   ├── face.html
│   └── sign_language.html
├── requirements.txt
├── config.py
└── README.md
```

## Features

- **Batik Generation**: Generate unique batik designs using GAN.
- **Face Processing**: Process and analyze facial images for various applications.
- **Sign Language Recognition**: Recognize and process sign language gestures.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd trunojoyo-ai
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

## Usage

- Access the main page at `http://localhost:8000/` to navigate through the features.
- Use the navigation menu to explore Batik generation, Face processing, and Sign Language recognition functionalities.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.