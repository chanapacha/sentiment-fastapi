# Sentiment Analysis Web Application
Created by  Chanapa Chareesan
## Overview
This project is a web application built with FastAPI that provides sentiment analysis functionality. It allows users to input text or upload an Excel file containing text data for sentiment analysis. The application analyzes the sentiment of the provided text data and returns the results to the user.

# Components
1. FastAPI Backend: Handles HTTP requests and serves the sentiment analysis endpoints.
2. HTML Interface: Provides a user-friendly interface for interacting with the sentiment analysis functionality.
3. Sentiment Analyzer Module: Contains the sentiment analysis logic used by the backend to classify text sentiment.
## Features
- Analyze sentiment of text input.
- Analyze sentiment of text data uploaded via Excel file.
- User-friendly interface for inputting text and viewing analysis results.
1. Usage
  - Text Input:
  - Enter text in the input box provided.
  - Click the "Analyze" button to perform sentiment analysis.
  - View the sentiment prediction, confidence score, and trigger status.
2. File Upload:
  - Select an Excel file containing text data.
  - Click the "Analyze Excel" button to perform sentiment analysis on the text data.
  - Download the analyzed Excel file with sentiment analysis results.
# Setup
1. Clone Repository: Clone the project repository from GitHub.
2. Install Dependencies: Install the required Python packages using pip.
3. Run the Application: Start the FastAPI application to run the web server.
4. Access the Interface: Open a web browser and navigate to the provided URL to access the sentiment analysis interface.
# Dependencies
- FastAPI: Web framework for building APIs with Python.
- Pydantic: Data validation and settings management using Python type annotations.
- Pandas: Data manipulation and analysis library for Python.
- Sentiment Analyzer Module: Custom module containing sentiment analysis logic.
# Future Enhancements
- Improve error handling and validation for input data.
- Enhance the user interface with additional features and styling.
- Integrate with a database for storing analysis results and user data.
- Support additional file formats for data upload.
