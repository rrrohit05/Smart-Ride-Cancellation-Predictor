from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()

# Load model and encoder
model = joblib.load("models/cancel_model.pkl")
encoder = joblib.load("models/encoder.pkl")

templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    vehicle_type: str = Form(...),
    pickup_location: str = Form(...),
    drop_location: str = Form(...),
    payment_method: str = Form(...),
    booking_day: int = Form(...),
    booking_hour: int = Form(...),
):
    input_data = pd.DataFrame([{
        "Vehicle Type": vehicle_type,
        "Pickup Location": pickup_location,
        "Drop Location": drop_location,
        "Payment Method": payment_method,
        "booking_day": booking_day,
        "booking_hour": booking_hour
    }])

    encoded_data = encoder.transform(input_data)

    prob = model.predict_proba(encoded_data)[0][1]
    prediction = model.predict(encoded_data)[0]
    print("INPUT DATA:")
    print(input_data)
    print("Encoded shape:", encoded_data.shape)
    print("Probability:", prob)
    print("Encoded row sum:", encoded_data.sum())
    result = "Cancelled" if prediction == 1 else "Success"
    confidence = round(prob * 100, 2)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": result,
            "confidence": confidence
        }
    ) 5
