# api/views.py
import os
import tempfile

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# Import predictor functions (your placeholder files in api/utils/)
from .utils.k_predict_image import predict_image
from .utils.k_predict_text import predict_text
from .utils.t_predict_video import predict_video
from .utils.t_predict_audio import predict_audio


def format_response(result, confidence):
    """Uniform JSON response format."""
    try:
        conf = float(confidence)
    except Exception:
        conf = 0.0
    return {"result": result, "confidence": round(conf, 2)}


@api_view(['GET'])
def ping(request):
    """Simple health check."""
    return Response({"message": "Backend is active!"}, status=status.HTTP_200_OK)


def save_uploaded_file(uploaded_file, suffix=""):
    """
    Save an uploaded file to a temporary file and return its path.
    uploaded_file: request.FILES['file']
    suffix: file extension or suffix (e.g. '.jpg', '.mp4', '.wav')
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        return tmp.name


@api_view(['POST'])
def analyze_image(request):
    """
    POST /api/analyze_image/
    form-data: key = file (type: File)
    """
    uploaded_file = request.FILES.get('file')
    if not uploaded_file:
        return Response({"error": "No file uploaded. Use key 'file' in form-data."},
                        status=status.HTTP_400_BAD_REQUEST)

    # Optional: basic file size check (10MB)
    max_size = 10 * 1024 * 1024
    if uploaded_file.size > max_size:
        return Response({"error": "File too large (max 10MB)."}, status=status.HTTP_400_BAD_REQUEST)

    # Save to temp and call predictor
    try:
        tmp_path = save_uploaded_file(uploaded_file, suffix=os.path.splitext(uploaded_file.name)[1] or ".img")
        result, confidence = predict_image(tmp_path)
    except Exception as e:
        return Response({"error": "Prediction error", "detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        # Optionally remove temp file if you don't need it after inference
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return Response(format_response(result, confidence), status=status.HTTP_200_OK)


@api_view(['POST'])
def analyze_video(request):
    """
    POST /api/analyze_video/
    form-data: key = file (type: File)
    """
    uploaded_file = request.FILES.get('file')
    if not uploaded_file:
        return Response({"error": "No file uploaded. Use key 'file'."}, status=status.HTTP_400_BAD_REQUEST)

    # Basic size check (e.g., 100MB)
    max_size = 100 * 1024 * 1024
    if uploaded_file.size > max_size:
        return Response({"error": "File too large (max 100MB)."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        tmp_path = save_uploaded_file(uploaded_file, suffix=os.path.splitext(uploaded_file.name)[1] or ".mp4")
        result, confidence = predict_video(tmp_path)
    except Exception as e:
        return Response({"error": "Prediction error", "detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return Response(format_response(result, confidence), status=status.HTTP_200_OK)


@api_view(['POST'])
def analyze_audio(request):
    """
    POST /api/analyze_audio/
    form-data: key = file (type: File)
    """
    uploaded_file = request.FILES.get('file')
    if not uploaded_file:
        return Response({"error": "No file uploaded. Use key 'file'."}, status=status.HTTP_400_BAD_REQUEST)

    # Basic size check (e.g., 20MB)
    max_size = 20 * 1024 * 1024
    if uploaded_file.size > max_size:
        return Response({"error": "File too large (max 20MB)."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        tmp_path = save_uploaded_file(uploaded_file, suffix=os.path.splitext(uploaded_file.name)[1] or ".wav")
        result, confidence = predict_audio(tmp_path)
    except Exception as e:
        return Response({"error": "Prediction error", "detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return Response(format_response(result, confidence), status=status.HTTP_200_OK)


@api_view(['POST'])
def analyze_text(request):
    """
    POST /api/analyze_text/
    JSON body: { "text": "..." }
    """
    data = request.data
    text_input = data.get('text')
    if not text_input:
        return Response({"error": "No 'text' provided in JSON body."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        result, confidence = predict_text(text_input)
    except Exception as e:
        return Response({"error": "Prediction error", "detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(format_response(result, confidence), status=status.HTTP_200_OK)
