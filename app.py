import streamlit as st
import requests
import json
from pypdf import PdfReader
from io import BytesIO
import smtplib # Added for email functionality
from email.mime.text import MIMEText # Added for email functionality

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# --- AGENT DEFINITIONS ---

class DataExtractorAgent:
    """
    A specialized agent responsible for extracting structured data from document text
    using an LLM.
    """
    def run(self, document_text: str) -> dict:
        """
        Processes the document text to extract key information.

        Args:
            document_text: The text content of the resume.

        Returns:
            A dictionary containing the extracted data or None if an error occurs.
        """
        st.info(" L1 Agent (Data Extractor): Task received. Analyzing document...")
        prompt = f"""
        You are an expert HR assistant specializing in parsing resumes for background verification.
        Extract the following information from the provided resume text:
        - full_name
        - email_address
        - phone_number
        - list_of_universities
        - list_of_previous_employers

        Here is the resume text:
        ---
        {document_text}
        ---

        Please provide the output ONLY as a valid JSON object. Do not include any other text, explanations, or markdown.
        If a piece of information is not found, use a null value or an empty list.
        """
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
            response.raise_for_status()

            response_data = response.json()
            extracted_json = json.loads(response_data.get("response", "{}"))
            st.success("L1 Agent (Data Extractor): Data extraction successful.")
            return extracted_json

        except requests.exceptions.RequestException as e:
            st.error(f"L1 Agent (Data Extractor): Error connecting to Ollama: {e}")
            return None
        except json.JSONDecodeError:
            st.error("L1 Agent (Data Extractor): Failed to parse the model's response as JSON.")
            st.code(response.text, language="text")
            return None

class VerificationMailerAgent:
    """
    A specialized agent responsible for sending verification emails.
    """
    def run(self, extracted_data: dict) -> str:
        """
        Sends a verification email based on the extracted data.
        NOTE: This is a simulation and requires real SMTP configuration to send actual emails.

        Args:
            extracted_data: A dictionary with candidate information.

        Returns:
            A string indicating the status of the email operation.
        """
        st.info("L1 Agent (Mailer): Task received. Preparing verification email...")
        full_name = extracted_data.get("full_name")
        email_address = extracted_data.get("email_address")

        if not all([full_name, email_address]):
            status = "Could not send email: 'full_name' or 'email_address' is missing."
            st.warning(f"L1 Agent (Mailer): {status}")
            return status

        # --- SMTP Configuration (Replace with your actual details) ---
        # For a real application, use st.secrets for this information
        SMTP_SERVER = "smtp.example.com"
        SMTP_PORT = 587
        SMTP_USERNAME = "your_username"
        SMTP_PASSWORD = "your_password"
        SENDER_EMAIL = "no-reply@yourcompany.com"

        # Create email body
        subject = f"Action Required: Background Information Verification for {full_name}"
        body = f"""
        Dear {full_name},

        As part of our recruitment process, we need to verify the information you provided.
        Please review the details below:

        - Email: {email_address}
        - Phone: {extracted_data.get("phone_number", "N/A")}
        - Universities: {', '.join(extracted_data.get("list_of_universities", []))}
        - Previous Employers: {', '.join(extracted_data.get("list_of_previous_employers", []))}

        If any of this information is incorrect, please reply to this email.

        Best regards,
        The HR Team
        """
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = email_address

        # --- SIMULATION MODE ---
        # In a real-world scenario, you would uncomment the block below to send emails.
        # try:
        #     with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        #         server.starttls()
        #         server.login(SMTP_USERNAME, SMTP_PASSWORD)
        #         server.send_message(msg)
        #         status = f"Successfully sent verification email to {email_address}."
        #         st.success(f"L1 Agent (Mailer): {status}")
        # except Exception as e:
        #     status = f"Failed to send email: {e}"
        #     st.error(f"L1 Agent (Mailer): {status}")
        # return status

        # For this example, we will just simulate the action and display the email content.
        st.success("L1 Agent (Mailer): Email prepared successfully (Simulation).")
        st.subheader("Simulated Email Content:")
        st.text(f"To: {email_address}")
        st.text(f"Subject: {subject}")
        st.text_area("Body", body, height=300)
        return f"Simulated sending of verification email to {email_address}."


class HRManagerAgent:
    """
    The orchestrator agent that manages the entire background verification workflow.
    """
    def __init__(self):
        self.extractor = DataExtractorAgent()
        self.mailer = VerificationMailerAgent()

    def run_workflow(self, document_text: str):
        """
        Executes the full workflow from data extraction to sending a verification email.

        Args:
            document_text: The text content of the resume.
        """
        st.header("🚀 Agent Workflow Initialized")
        st.info("HR Manager: Workflow started. Delegating to Data Extractor Agent.")

        # Step 1: Call the Data Extractor Agent
        extracted_data = self.extractor.run(document_text)

        if not extracted_data:
            st.error("HR Manager: Data extraction failed. Halting workflow.")
            return

        st.subheader("✅ Extracted Information")
        st.json(extracted_data)
        st.session_state['extracted_data'] = extracted_data # Save for later use

        # Step 2: Call the Verification Mailer Agent
        st.info("HR Manager: Extraction complete. Delegating to Verification Mailer Agent.")
        mail_status = self.mailer.run(extracted_data)
        st.session_state['mail_status'] = mail_status

        st.success("HR Manager: All tasks completed. Workflow finished.")


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI Background Verification Agent")

st.title("📄 Automated Background Verification Agent")
st.write("Upload a candidate's resume (PDF) to automatically extract key information and initiate verification.")

# Initialize the manager agent
manager_agent = HRManagerAgent()

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.success(f"Successfully uploaded: **{uploaded_file.name}**")

    if st.button("🔎 Start Agent Workflow", type="primary"):
        with st.spinner("The AI agents are processing the document..."):
            try:
                bytes_data = uploaded_file.getvalue()
                pdf_file = BytesIO(bytes_data)
                reader = PdfReader(pdf_file)

                document_text = ""
                for page in reader.pages:
                    document_text += page.extract_text() + "\n"

                if not document_text.strip():
                    st.error("Could not extract any text from the PDF. The file might be an image-based PDF.")
                else:
                    # The HR Manager handles the entire process
                    manager_agent.run_workflow(document_text)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")