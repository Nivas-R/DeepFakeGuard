# api/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import tempfile

# A small test route to confirm your API is running
@api_view(['GET'])
def ping(request):
    return Response({"message": "Backend is active!"}, status=status.HTTP_200_OK)


# Main dummy endpoint
@api_view(['POST'])
def analyze_image(request):
    """
    Accepts an uploaded image file and returns a fake deepfake result.
    """
    # Step 1: Get the file from request
    uploaded_file = request.FILES.get('file')
    if not uploaded_file:
        return Response({"error": "No file uploaded. Please upload with key 'file'."},
                        status=status.HTTP_400_BAD_REQUEST)

    # Step 2: Save file temporarily (for now)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
        for chunk in uploaded_file.chunks():
            temp.write(chunk)
        file_path = temp.name

    # Step 3: Fake result (you’ll replace this with ML model later)
    result = "Fake"
    confidence = 0.75

    # Step 4: Return JSON response
    return Response({
        "filename": uploaded_file.name,
        "result": result,
        "confidence": confidence,
        "file_path": file_path
    }, status=status.HTTP_200_OK)
