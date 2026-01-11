"""
API Tests for SkyPredict
"""

import pytest
from fastapi.testclient import TestClient
from datetime import date


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "SkyPredict API"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_info" in data

    def test_readiness_endpoint(self, client):
        """Test readiness probe."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data

    def test_liveness_endpoint(self, client):
        """Test liveness probe."""
        response = client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_get_airports(self, client):
        """Test airports list endpoint."""
        response = client.get("/api/v1/predictions/airports")
        assert response.status_code == 200
        data = response.json()
        assert "airports" in data
        assert isinstance(data["airports"], list)
        assert len(data["airports"]) > 0

    def test_get_airlines(self, client):
        """Test airlines list endpoint."""
        response = client.get("/api/v1/predictions/airlines")
        assert response.status_code == 200
        data = response.json()
        assert "airlines" in data
        assert isinstance(data["airlines"], list)
        assert len(data["airlines"]) > 0

    def test_single_prediction_validation(self, client):
        """Test input validation for single prediction."""
        # Missing required fields
        response = client.post(
            "/api/v1/predictions/single",
            json={"airline": "AA"}
        )
        assert response.status_code == 422  # Validation error

    def test_single_prediction_invalid_airport(self, client):
        """Test validation for invalid airport code."""
        response = client.post(
            "/api/v1/predictions/single",
            json={
                "airline": "AA",
                "origin_airport": "XX",  # Invalid
                "dest_airport": "LAX",
                "flight_date": "2024-07-15",
                "scheduled_departure_hour": 14
            }
        )
        # Should still process (model handles unknown airports)
        # or return validation error depending on implementation
        assert response.status_code in [200, 422, 503]


class TestBatchPrediction:
    """Tests for batch prediction endpoint."""

    def test_batch_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post(
            "/api/v1/predictions/batch",
            json={"flights": []}
        )
        # Empty list should be rejected by validation
        assert response.status_code == 422

    def test_batch_too_many_flights(self, client):
        """Test batch prediction with too many flights."""
        flights = [
            {
                "airline": "AA",
                "origin_airport": "JFK",
                "dest_airport": "LAX",
                "flight_date": "2024-07-15",
                "scheduled_departure_hour": 14
            }
        ] * 101  # Exceeds max of 100

        response = client.post(
            "/api/v1/predictions/batch",
            json={"flights": flights}
        )
        assert response.status_code == 422  # Validation error


# Pytest fixtures
@pytest.fixture
def client():
    """Create test client."""
    from src.api.main import app
    return TestClient(app)


@pytest.fixture
def sample_flight():
    """Sample flight data for testing."""
    return {
        "airline": "AA",
        "flight_number": 1234,
        "origin_airport": "JFK",
        "dest_airport": "LAX",
        "flight_date": str(date.today()),
        "scheduled_departure_hour": 14,
        "scheduled_departure_minute": 30,
        "distance": 2475,
        "scheduled_duration": 330
    }
