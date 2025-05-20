# Dia-TTS Server

A powerful Text-to-Speech server with advanced features for generating audio content and creating short-form videos. This server is built using FastAPI and Modal, providing a scalable and efficient solution for TTS generation and content creation.

## Features

- **Text-to-Speech Generation**: Convert text to natural-sounding speech with customizable parameters
- **Voice Cloning**: Support for predefined voices and voice cloning capabilities
- **Complete Workflow Automation**: End-to-end pipeline for creating short-form videos including:
  - Story generation
  - Background music selection
  - Scene-based audio generation
  - Image selection and mapping
  - Video rendering with audio-visual synchronization

## Technical Stack

- **Framework**: FastAPI
- **Cloud Platform**: Modal
- **GPU Support**: NVIDIA L40S
- **Key Dependencies**:
  - PyTorch & TorchAudio
  - OpenAI Whisper
  - FastAPI
  - MoviePy
  - AWS S3 Integration
  - MongoDB Integration

## API Endpoints

### 1. Health Check
```
GET /
```
Returns the server status.

### 2. Simple TTS
```
POST /simple-tts
```
Generate speech from text with customizable parameters:
- `text`: Input text to convert to speech
- `speed`: Speech speed factor (default: 1.0)
- `seed`: Random seed for generation (default: 42)
- `voice`: Voice selection (default: "default")

### 3. Generate Short
```
GET /generate-short
```
Triggers the complete workflow for generating a short-form video.

## Complete Workflow Process

The server implements a comprehensive workflow for creating short-form videos:

1. **Story Generation**: Creates a narrative script
2. **Background Music**: Selects appropriate background music
3. **Scene Processing**: Generates audio for each scene
4. **Image Selection**: Maps relevant images to scenes
5. **Asset Management**: Generates pre-signed URLs for all assets
6. **Video Rendering**: Combines audio and images into a final video
7. **Output Storage**: Uploads the final video to S3

## Setup and Installation

1. Ensure you have Modal CLI installed and configured
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the necessary environment variables:
- AWS credentials for S3 access
- MongoDB connection details
- Other configuration parameters

## Usage

1. Start the server:
```bash
modal serve inference.py
```

2. Access the API endpoints using the provided Modal URL

## Architecture

The server is built with scalability in mind:
- Uses Modal for cloud deployment
- Implements GPU acceleration for TTS generation
- Integrates with S3 for asset storage
- Uses MongoDB for data persistence
- Implements efficient audio processing and video rendering

## Contributing

Please read the contribution guidelines before submitting pull requests.

## License

[Specify your license here]

## Contact

[Add contact information]
