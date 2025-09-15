# Data Explorer with Natural Commands

An AI-powered data exploration tool that allows users to analyze CSV data through natural language queries.

## Features

- **Single CSV File Upload**: Upload and analyze one CSV file at a time
- **Natural Language Queries**: Ask questions about your data in plain English
- **AI-Powered Analysis**: 
  - Get responses in tabular format
  - Request natural language explanations of the data
- **Export Options**: 
  - Export results to CSV
  - Export results to JSON
- **Interactive Interface**: Built with Streamlit for a seamless user experience

## Workspace Structure

- `Client/`  
  Contains the main application code.
  - `app.py` — Main application script.
  - `.env`, `.env.example` — Environment variable configuration files.

- `Project5.csv`  
  Sample dataset for exploration.

## Getting Started

1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Set up environment variables**  
   Copy `.env.example` to `.env` and update as needed.

3. **Install dependencies**  
   (If using Python, inside `Client/`)
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```sh
   streamlit run Client/app.py
   ```

## Usage

1. Launch the Streamlit app
2. Upload your CSV file using the file uploader
3. Ask questions about your data in natural language
4. View results in tabular format
5. Request explanations of the results if needed
6. Export results in CSV or JSON format

## License

MIT License