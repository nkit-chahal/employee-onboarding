import io
import os
import json
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, ValidationError
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from dotenv import load_dotenv

from pypdf import PdfReader
from openai import AzureOpenAI


# ===================== Pydantic Schemas (match your payload) =====================
class DocumentRef(BaseModel):
    fileName: str = Field(..., min_length=1)

class Documents(BaseModel):
    tenthMarksheet: Optional[DocumentRef] = None
    twelfthMarksheet: Optional[DocumentRef] = None
    bachelorsDegree: Optional[DocumentRef] = None
    bachelorsResult: Optional[DocumentRef] = None
    resume: Optional[DocumentRef] = None
    identityProof: Optional[DocumentRef] = None
    policeVerification: Optional[DocumentRef] = None
    aadhaarOrDomicile: Optional[DocumentRef] = None
    relievingLetter: Optional[List[DocumentRef]] = None
    salarySlips: Optional[List[DocumentRef]] = None
    otherCertificates: List[DocumentRef] = Field(default_factory=list)

class Metadata(BaseModel):
    candidateName: Optional[str] = None
    city: Optional[str] = None
    localAddress: Optional[str] = None
    permanentAddress: Optional[str] = None
    phoneNumber: Optional[str] = None
    email: Optional[EmailStr] = None
    employer: Optional[str] = None
    previousHrEmail: Optional[EmailStr] = None

class EmployeePayload(BaseModel):
    metadata: Metadata
    documents: Documents


# ===================== App / Config =====================
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "employee_db")
COLLECTION = "employees"

# Azure OpenAI
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g., "gpt-4o-mini" or your custom name

if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
    # We'll still start, but /ingest-files will fail if these aren't set.
    pass

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

app = FastAPI(title="Employee Data API + PDF LLM Parser", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ===================== Mongo Client =====================
@app.on_event("startup")
async def startup():
    app.state.mongo = AsyncIOMotorClient(MONGODB_URI)
    app.state.db = app.state.mongo[MONGODB_DB]
    await app.state.db[COLLECTION].create_index("metadata.email", unique=True, sparse=True)
    await app.state.db[COLLECTION].create_index("metadata.phoneNumber")

@app.on_event("shutdown")
async def shutdown():
    app.state.mongo.close()


# ===================== Utils =====================
def to_object_id(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId")

def serialize_doc(doc: dict) -> dict:
    if not doc:
        return doc
    doc["_id"] = str(doc["_id"])
    return doc

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF (native PDFs; for scans you'd add OCR later)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    raw = "\n".join(texts)
    # normalize whitespace a bit
    return re.sub(r"[ \t]+", " ", raw).strip()

def safe_trim(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    # trim at a boundary
    return text[:max_chars] + "\n...[TRIMMED]"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\d{10})")

def simple_fallback(texts_merged: str) -> Dict[str, Optional[str]]:
    email = None
    phone = None
    m = EMAIL_RE.search(texts_merged)
    if m:
        email = m.group(0)
    m2 = PHONE_RE.search(texts_merged.replace(" ", ""))
    if m2:
        phone = m2.group(0)[-10:]  # last 10 digits for Indian numbers
    return {"email": email, "phoneNumber": phone}

def build_llm_prompt(doc_chunks: Dict[str, str]) -> list:
    """Create a ChatML message list for Azure OpenAI parsing."""
    schema = {
        "metadata": {
            "candidateName": "string or null",
            "city": "string or null",
            "localAddress": "string or null",
            "permanentAddress": "string or null",
            "phoneNumber": "string or null",
            "email": "string or null",
            "employer": "string or null",
            "previousHrEmail": "string or null"
        },
        "documents": {
            "tenthMarksheet": {"fileName": "string"} if "tenthMarksheet" in doc_chunks else None,
            "twelfthMarksheet": {"fileName": "string"} if "twelfthMarksheet" in doc_chunks else None,
            "bachelorsDegree": {"fileName": "string"} if "bachelorsDegree" in doc_chunks else None,
            "bachelorsResult": {"fileName": "string"} if "bachelorsResult" in doc_chunks else None,
            "resume": {"fileName": "string"} if "resume" in doc_chunks else None,
            "identityProof": {"fileName": "string"} if "identityProof" in doc_chunks else None,
            "policeVerification": {"fileName": "string"} if "policeVerification" in doc_chunks else None,
            "aadhaarOrDomicile": {"fileName": "string"} if "aadhaarOrDomicile" in doc_chunks else None,
            "relievingLetter": [{"fileName": "string"}] if "relievingLetter" in doc_chunks else None,
            "salarySlips": [{"fileName": "string"}] if "salarySlips" in doc_chunks else None,
            "otherCertificates": [{"fileName": "string"}] if "otherCertificates" in doc_chunks else []
        }
    }

    # System + instructions
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise information extraction assistant. "
                "Given labeled excerpts from candidate documents (PDF text), "
                "return a SINGLE JSON object matching the schema below. "
                "If a field is not found, return null (for scalar fields) or omit arrays. "
                "Never invent facts. Infer cautiously when obvious (e.g., same name repeated). "
                "IMPORTANT: Output ONLY valid JSON (no comments, no markdown)."
            ),
        },
        {
            "role": "user",
            "content": (
                "JSON schema example (keys present even if null for scalars):\n"
                + json.dumps(schema, indent=2)
            ),
        },
    ]

    # Add labeled doc chunks
    labeled_blobs = []
    for key, text in doc_chunks.items():
        labeled_blobs.append(f"=== {key} START ===\n{text}\n=== {key} END ===")
    messages.append({
        "role": "user",
        "content": "Here are the labeled document texts:\n\n" + "\n\n".join(labeled_blobs)
    })

    # Final reminder
    messages.append({
        "role": "user",
        "content": (
            "Return JSON matching the schema with best-effort values for metadata "
            "(candidateName, email, phoneNumber, addresses, city, employer, previousHrEmail) "
            "derived across ALL documents. For documents arrays, use the uploaded filename(s) only."
        ),
    })
    return messages

async def upsert_employee(db, employee: EmployeePayload) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    body = employee.model_dump()
    # Upsert by email if present; otherwise insert a new doc
    filter_ = {"metadata.email": employee.metadata.email} if employee.metadata.email else {"_id": ObjectId()}
    result = await db[COLLECTION].update_one(
        filter_,
        {"$set": {**body, "updatedAt": now}, "$setOnInsert": {"createdAt": now}},
        upsert=True,
    )
    if result.upserted_id:
        oid = result.upserted_id
        action = "inserted"
    else:
        doc = await db[COLLECTION].find_one(filter_, {"_id": 1})
        oid = doc["_id"]
        action = "updated"
    return {"status": "ok", "action": action, "id": str(oid)}


# ===================== Routes =====================
@app.get("/health")
async def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.post("/upload")
async def upload(payload: EmployeePayload):
    """
    Original JSON upsert (no PDF parsing).
    """
    res = await upsert_employee(app.state.db, payload)
    return res


@app.post("/ingest-files")
async def ingest_files(
    # Parallel lists: one doc_type per file (same order)
    doc_types: List[str] = Form(..., description="Doc type per file (e.g., resume, tenthMarksheet, salarySlips)"),
    files: List[UploadFile] = File(..., description="PDF files aligned with doc_types"),
):
    """
    Accepts multiple PDFs, extracts text via pypdf, uses Azure OpenAI to parse a structured EmployeePayload,
    and upserts into MongoDB. You choose doc_types for each file (same order as files).
    Supported doc_types include:
      tenthMarksheet, twelfthMarksheet, bachelorsDegree, bachelorsResult, resume,
      identityProof, policeVerification, aadhaarOrDomicile, relievingLetter,
      salarySlips, otherCertificates
    """
    if AZURE_API_KEY is None or AZURE_ENDPOINT is None or AZURE_DEPLOYMENT is None:
        raise HTTPException(status_code=500, detail="Azure OpenAI env vars not configured")

    if len(doc_types) != len(files):
        raise HTTPException(status_code=400, detail="doc_types length must match files length")

    # 1) Extract PDF text and prepare doc mapping for LLM
    doc_texts: Dict[str, str] = {}
    filename_map: Dict[str, List[str]] = {}

    for dt, f in zip(doc_types, files):
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF")
        raw_bytes = await f.read()
        text = extract_pdf_text(raw_bytes)
        text = safe_trim(text, 12000)

        # For multi-valued docs (relievingLetter, salarySlips, otherCertificates), collect as arrays
        if dt in ("relievingLetter", "salarySlips", "otherCertificates"):
            # concatenate texts for LLM, but keep per-file filenames
            doc_texts[dt] = (doc_texts.get(dt, "") + "\n" + text).strip()
            filename_map.setdefault(dt, []).append(f.filename)
        else:
            # single-valued docs: last one wins if provided multiple times
            doc_texts[dt] = text
            filename_map[dt] = [f.filename]

    # 2) LLM parsing
    messages = build_llm_prompt(doc_texts)
    try:
        completion = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            temperature=0,
            max_tokens=1200,
            messages=messages,
        )
        content = completion.choices[0].message.content.strip()
        # Sometimes models wrap with ```json ... ```
        content = content.strip().strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
        parsed = json.loads(content)
    except Exception as e:
        # Fallback: minimal object with just metadata from regex if possible
        merged_text = "\n".join(doc_texts.values())
        fb = simple_fallback(merged_text)
        parsed = {
            "metadata": {
                "candidateName": None,
                "city": None,
                "localAddress": None,
                "permanentAddress": None,
                "phoneNumber": fb.get("phoneNumber"),
                "email": fb.get("email"),
                "employer": None,
                "previousHrEmail": None,
            },
            "documents": {},
        }

    # 3) Attach filenames into documents (LLM should return schema keys; we enforce)
    docs: Dict[str, Any] = parsed.get("documents", {}) or {}
    # Ensure all present doc types are reflected with fileName(s)
    for key, names in filename_map.items():
        if key in ("relievingLetter", "salarySlips", "otherCertificates"):
            docs[key] = [{"fileName": n} for n in names]
        else:
            docs[key] = {"fileName": names[-1]}  # single ref

    parsed["documents"] = docs

    # Ensure metadata keys exist even if null
    meta = parsed.get("metadata") or {}
    for k in ["candidateName", "city", "localAddress", "permanentAddress",
              "phoneNumber", "email", "employer", "previousHrEmail"]:
        meta.setdefault(k, None)
    parsed["metadata"] = meta

    # 4) Validate against Pydantic schema
    try:
        employee = EmployeePayload(**parsed)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=json.loads(ve.json()))

    # 5) Upsert into Mongo
    res = await upsert_employee(app.state.db, employee)
    return {
        **res,
        "parsedMetadata": employee.metadata.model_dump(),
        "documentsStored": list(filename_map.keys()),
    }


@app.get("/employees/{id}")
async def get_employee(id: str):
    doc = await app.state.db[COLLECTION].find_one({"_id": to_object_id(id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return serialize_doc(doc)

@app.get("/employees")
async def list_employees(email: Optional[EmailStr] = None, limit: int = 50):
    query = {"metadata.email": str(email)} if email else {}
    cursor = app.state.db[COLLECTION].find(query).limit(min(max(limit, 1), 200))
    items = [serialize_doc(d) async for d in cursor]
    return {"count": len(items), "items": items}
