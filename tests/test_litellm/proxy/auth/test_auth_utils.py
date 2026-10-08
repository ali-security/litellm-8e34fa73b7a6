"""
Unit tests for auth_utils functions related to rate limiting and customer ID extraction.
"""

import json
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import (
    _get_customer_id_from_standard_headers,
    check_complete_credentials,
    get_end_user_id_from_request_body,
    get_model_from_request,
    get_key_model_rpm_limit,
    get_key_model_tpm_limit,
    get_project_model_rpm_limit,
    get_project_model_tpm_limit,
    is_request_body_safe,
)


class TestGetKeyModelRpmLimit:
    """Tests for get_key_model_rpm_limit function."""

    def test_returns_key_metadata_when_present(self):
        """Key metadata takes priority over team metadata."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={"model_rpm_limit": {"gpt-4": 100}},
            team_metadata={"model_rpm_limit": {"gpt-4": 50}},
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 100}

    def test_falls_back_to_team_metadata_when_key_has_other_metadata(self):
        """Should fall back to team metadata when key metadata exists but has no model_rpm_limit."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={
                "some_other_key": "value"
            },  # Has metadata, but not model_rpm_limit
            team_metadata={"model_rpm_limit": {"gpt-4": 50}},
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 50}

    def test_extracts_from_model_max_budget(self):
        """Should extract rpm_limit from model_max_budget when metadata is empty."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"rpm_limit": 100, "tpm_limit": 1000},
                "gpt-3.5-turbo": {"rpm_limit": 200},
            },
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 100, "gpt-3.5-turbo": 200}

    def test_skips_models_without_rpm_limit(self):
        """Should skip models that don't have rpm_limit in model_max_budget."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"rpm_limit": 100},
                "gpt-3.5-turbo": {"tpm_limit": 1000},  # No rpm_limit
            },
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 100}

    def test_returns_none_when_no_limits_configured(self):
        """Should return None when no rate limits are configured."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result is None

    def test_team_metadata_empty_rpm_dict_falls_through_to_deployment_default(self):
        """Explicitly empty team model_rpm_limit ({}) should be returned as-is, not fallen through."""
        # An empty dict is a valid team limit map (no per-model limits configured).
        # It should be returned directly rather than falling through to deployment defaults,
        # so a team with an empty map is treated as unconstrained at the team level.
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            team_metadata={"model_rpm_limit": {}},
        )
        result = get_key_model_rpm_limit(user_api_key_dict)
        assert result == {}


class TestGetKeyModelTpmLimit:
    """Tests for get_key_model_tpm_limit function."""

    def test_returns_key_metadata_when_present(self):
        """Key metadata takes priority over team metadata."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={"model_tpm_limit": {"gpt-4": 10000}},
            team_metadata={"model_tpm_limit": {"gpt-4": 5000}},
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000}

    def test_falls_back_to_team_metadata_when_key_has_other_metadata(self):
        """Should fall back to team metadata when key metadata exists but has no model_tpm_limit."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={
                "some_other_key": "value"
            },  # Has metadata, but not model_tpm_limit
            team_metadata={"model_tpm_limit": {"gpt-4": 5000}},
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 5000}

    def test_extracts_from_model_max_budget(self):
        """Should extract tpm_limit from model_max_budget when metadata is empty."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"tpm_limit": 10000, "rpm_limit": 100},
                "gpt-3.5-turbo": {"tpm_limit": 20000},
            },
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000, "gpt-3.5-turbo": 20000}

    def test_skips_models_without_tpm_limit(self):
        """Should skip models that don't have tpm_limit in model_max_budget."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={
                "gpt-4": {"tpm_limit": 10000},
                "gpt-3.5-turbo": {"rpm_limit": 100},  # No tpm_limit
            },
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000}

    def test_returns_none_when_no_limits_configured(self):
        """Should return None when no rate limits are configured."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result is None

    def test_model_max_budget_priority_over_team(self):
        """model_max_budget should take priority over team_metadata."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            model_max_budget={"gpt-4": {"tpm_limit": 10000}},
            team_metadata={"model_tpm_limit": {"gpt-4": 5000}},
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 10000}

    def test_team_metadata_empty_tpm_dict_falls_through_to_deployment_default(self):
        """Explicitly empty team model_tpm_limit ({}) should be returned as-is, not fallen through."""
        # An empty dict is a valid team limit map (no per-model limits configured).
        # It should be returned directly rather than falling through to deployment defaults,
        # so a team with an empty map is treated as unconstrained at the team level.
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            team_metadata={"model_tpm_limit": {}},
        )
        result = get_key_model_tpm_limit(user_api_key_dict)
        assert result == {}

    def test_skips_deployments_with_malformed_limit_value(self):
        """Deployments with non-integer-parseable limit values are skipped without raising."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "model1",
                "litellm_params": {"default_api_key_tpm_limit": "not-a-number"},
            },
            _make_deployment_dict("model1", tpm=500),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        # The malformed deployment is skipped; the valid one provides 500
        assert result == {"model1": 500}


class TestGetCustomerIdFromStandardHeaders:
    """Tests for _get_customer_id_from_standard_headers helper function."""

    def test_should_return_customer_id_from_x_litellm_customer_id_header(self):
        """Should extract customer ID from x-litellm-customer-id header."""
        headers = {"x-litellm-customer-id": "customer-123"}
        result = _get_customer_id_from_standard_headers(request_headers=headers)
        assert result == "customer-123"

    def test_should_return_customer_id_from_x_litellm_end_user_id_header(self):
        """Should extract customer ID from x-litellm-end-user-id header."""
        headers = {"x-litellm-end-user-id": "end-user-456"}
        result = _get_customer_id_from_standard_headers(request_headers=headers)
        assert result == "end-user-456"

    def test_should_return_none_when_headers_is_none(self):
        """Should return None when headers is None."""
        result = _get_customer_id_from_standard_headers(request_headers=None)
        assert result is None

    def test_should_return_none_when_no_standard_headers_present(self):
        """Should return None when no standard customer ID headers are present."""
        headers = {"x-other-header": "some-value"}
        result = _get_customer_id_from_standard_headers(request_headers=headers)
        assert result is None


class TestGetEndUserIdFromRequestBodyWithStandardHeaders:
    """Tests for get_end_user_id_from_request_body with standard customer ID headers."""

    def test_should_prioritize_standard_header_over_body_user(self):
        """Standard customer ID header should take precedence over body user field."""
        headers = {"x-litellm-customer-id": "header-customer"}
        request_body = {"user": "body-user"}

        with patch("litellm.proxy.proxy_server.general_settings", {}):
            result = get_end_user_id_from_request_body(
                request_body=request_body, request_headers=headers
            )
        assert result == "header-customer"

    def test_should_fall_back_to_body_when_no_standard_header(self):
        """Should fall back to body user when no standard headers are present."""
        headers = {"x-other-header": "value"}
        request_body = {"user": "body-user"}

        with patch("litellm.proxy.proxy_server.general_settings", {}):
            result = get_end_user_id_from_request_body(
                request_body=request_body, request_headers=headers
            )
        assert result == "body-user"


def test_get_model_from_request_supports_google_model_names_with_slashes():
    assert (
        get_model_from_request(
            request_data={},
            route="/v1beta/models/bedrock/claude-sonnet-3.7:generateContent",
        )
        == "bedrock/claude-sonnet-3.7"
    )
    assert (
        get_model_from_request(
            request_data={},
            route="/models/hosted_vllm/gpt-oss-20b:generateContent",
        )
        == "hosted_vllm/gpt-oss-20b"
    )


def test_get_model_from_request_vertex_passthrough_still_works():
    route = "/vertex_ai/v1/projects/p/locations/l/publishers/google/models/gemini-1.5-pro:generateContent"
    assert get_model_from_request(request_data={}, route=route) == "gemini-1.5-pro"


def test_get_customer_user_header_returns_none_when_no_customer_role():
    from litellm.proxy.auth.auth_utils import get_customer_user_header_from_mapping

    mappings = [
        {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"}
    ]
    result = get_customer_user_header_from_mapping(mappings)
    assert result is None


def test_get_customer_user_header_returns_none_for_single_non_customer_mapping():
    from litellm.proxy.auth.auth_utils import get_customer_user_header_from_mapping

    mapping = {"header_name": "X-Only-Internal", "litellm_user_role": "internal_user"}
    result = get_customer_user_header_from_mapping(mapping)
    assert result is None


def test_get_customer_user_header_from_mapping_returns_customer_header():
    from litellm.proxy.auth.auth_utils import get_customer_user_header_from_mapping

    mappings = [
        {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"},
        {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"},
    ]
    result = get_customer_user_header_from_mapping(mappings)
    assert result == ["x-openwebui-user-email"]


def test_get_customer_user_header_returns_customers_header_in_config_order_when_multiple_exist():
    from litellm.proxy.auth.auth_utils import get_customer_user_header_from_mapping

    mappings = [
        {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"},
        {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"},
        {"header_name": "X-User-Id", "litellm_user_role": "customer"},
    ]
    result = get_customer_user_header_from_mapping(mappings)
    assert result == ["x-openwebui-user-email", "x-user-id"]


def test_get_end_user_id_returns_id_from_user_header_mappings():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    mappings = [
        {"header_name": "x-openwebui-user-id", "litellm_user_role": "internal_user"},
        {"header_name": "x-openwebui-user-email", "litellm_user_role": "customer"},
    ]
    general_settings = {"user_header_mappings": mappings}
    headers = {"x-openwebui-user-email": "1234"}

    with (
        patch(
            "litellm.proxy.auth.auth_utils._get_customer_id_from_standard_headers",
            return_value=None,
        ),
        patch("litellm.proxy.proxy_server.general_settings", general_settings),
    ):
        result = get_end_user_id_from_request_body(
            request_body={}, request_headers=headers
        )

    assert result == "1234"


def test_get_end_user_id_returns_first_customer_header_when_multiple_mappings_exist():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    mappings = [
        {"header_name": "x-openwebui-user-id", "litellm_user_role": "internal_user"},
        {"header_name": "x-user-id", "litellm_user_role": "customer"},
        {"header_name": "x-openwebui-user-email", "litellm_user_role": "customer"},
    ]
    general_settings = {"user_header_mappings": mappings}
    headers = {
        "x-user-id": "user-456",
        "x-openwebui-user-email": "user@example.com",
    }

    with (
        patch(
            "litellm.proxy.auth.auth_utils._get_customer_id_from_standard_headers",
            return_value=None,
        ),
        patch("litellm.proxy.proxy_server.general_settings", general_settings),
    ):
        result = get_end_user_id_from_request_body(
            request_body={}, request_headers=headers
        )

    assert result == "user-456"


def test_get_end_user_id_returns_none_when_no_customer_role_in_mappings():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    mappings = [
        {"header_name": "x-openwebui-user-id", "litellm_user_role": "internal_user"},
    ]
    general_settings = {"user_header_mappings": mappings}
    headers = {"x-openwebui-user-id": "user-789"}

    with (
        patch(
            "litellm.proxy.auth.auth_utils._get_customer_id_from_standard_headers",
            return_value=None,
        ),
        patch("litellm.proxy.proxy_server.general_settings", general_settings),
    ):
        result = get_end_user_id_from_request_body(
            request_body={}, request_headers=headers
        )

    assert result is None


def test_get_end_user_id_falls_back_to_deprecated_user_header_name():
    from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

    general_settings = {"user_header_name": "x-custom-user-id"}
    headers = {"x-custom-user-id": "user-legacy"}

    with (
        patch(
            "litellm.proxy.auth.auth_utils._get_customer_id_from_standard_headers",
            return_value=None,
        ),
        patch("litellm.proxy.proxy_server.general_settings", general_settings),
    ):
        result = get_end_user_id_from_request_body(
            request_body={}, request_headers=headers
        )

    assert result == "user-legacy"


def _make_deployment_dict(
    model_name: str, tpm: Optional[int] = None, rpm: Optional[int] = None
) -> dict:
    """Helper to build a minimal deployment dict as returned by router.get_model_list."""
    litellm_params: dict = {"model": model_name}
    if tpm is not None:
        litellm_params["default_api_key_tpm_limit"] = tpm
    if rpm is not None:
        litellm_params["default_api_key_rpm_limit"] = rpm
    return {"model_name": model_name, "litellm_params": litellm_params}


_ROUTER_PATCH = "litellm.proxy.proxy_server.llm_router"


class TestDeploymentDefaultRpmLimit:
    """Tests for deployment default_api_key_rpm_limit fallback in get_key_model_rpm_limit."""

    def test_returns_deployment_default_when_key_has_no_limits(self):
        """Case 2 from spec: key has no model-specific limits, falls back to deployment default."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", rpm=200)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 200}

    def test_key_model_limit_takes_priority_over_deployment_default(self):
        """Case 1 from spec: key model-specific limit wins over deployment default."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={"model_rpm_limit": {"model1": 10}},
        )
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", rpm=200)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 10}

    def test_returns_none_when_no_deployment_default_and_no_key_limits(self):
        """Returns None when neither the key nor the deployment has any rpm limit."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1")  # no rpm default
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result is None

    def test_returns_none_without_model_name_even_when_deployment_has_default(self):
        """No model_name means deployment fallback is skipped."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", rpm=200)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict)
        assert result is None

    def test_returns_none_when_llm_router_is_none(self):
        """No router means deployment fallback returns None gracefully."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        with patch(_ROUTER_PATCH, None):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result is None

    def test_returns_minimum_across_multiple_deployments(self):
        """When multiple deployments share a model name, the minimum rpm limit is used."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", rpm=200),
            _make_deployment_dict("model1", rpm=50),
            _make_deployment_dict("model1", rpm=150),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 50}

    def test_ignores_deployments_without_default_when_others_have_it(self):
        """Deployments missing the field are skipped; min is taken over those that have it."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1"),  # no rpm default
            _make_deployment_dict("model1", rpm=75),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 75}

    def test_skips_deployments_with_malformed_limit_value(self):
        """Deployments with non-integer-parseable limit values are skipped without raising."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            {
                "model_name": "model1",
                "litellm_params": {"default_api_key_rpm_limit": "not-a-number"},
            },
            _make_deployment_dict("model1", rpm=100),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_rpm_limit(user_api_key_dict, model_name="model1")
        # The malformed deployment is skipped; the valid one provides 100
        assert result == {"model1": 100}


class TestDeploymentDefaultTpmLimit:
    """Tests for deployment default_api_key_tpm_limit fallback in get_key_model_tpm_limit."""

    def test_returns_deployment_default_when_key_has_no_limits(self):
        """Case 2 from spec: key has no model-specific limits, falls back to deployment default."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", tpm=100)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 100}

    def test_key_model_limit_takes_priority_over_deployment_default(self):
        """Case 1 from spec: key model-specific limit wins over deployment default."""
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            metadata={"model_tpm_limit": {"model1": 20}},
        )
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", tpm=100)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 20}

    def test_returns_none_when_no_deployment_default_and_no_key_limits(self):
        """Returns None when neither the key nor the deployment has any tpm limit."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1")  # no tpm default
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result is None

    def test_returns_none_without_model_name_even_when_deployment_has_default(self):
        """No model_name means deployment fallback is skipped."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", tpm=100)
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict)
        assert result is None

    def test_returns_none_when_llm_router_is_none(self):
        """No router means deployment fallback returns None gracefully."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        with patch(_ROUTER_PATCH, None):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result is None

    def test_returns_minimum_across_multiple_deployments(self):
        """When multiple deployments share a model name, the minimum tpm limit is used."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1", tpm=1000),
            _make_deployment_dict("model1", tpm=300),
            _make_deployment_dict("model1", tpm=700),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 300}

    def test_ignores_deployments_without_default_when_others_have_it(self):
        """Deployments missing the field are skipped; min is taken over those that have it."""
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        mock_router = MagicMock()
        mock_router.get_model_list.return_value = [
            _make_deployment_dict("model1"),  # no tpm default
            _make_deployment_dict("model1", tpm=400),
        ]
        with patch(_ROUTER_PATCH, mock_router):
            result = get_key_model_tpm_limit(user_api_key_dict, model_name="model1")
        assert result == {"model1": 400}


class TestGetProjectModelRpmLimit:
    """Tests for get_project_model_rpm_limit function."""

    def test_returns_project_metadata_rpm_limit(self):
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            project_metadata={"model_rpm_limit": {"gpt-4": 200}},
        )
        result = get_project_model_rpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 200}

    def test_returns_none_when_no_project_metadata(self):
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        result = get_project_model_rpm_limit(user_api_key_dict)
        assert result is None

    def test_returns_none_when_project_metadata_missing_key(self):
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            project_metadata={"other_key": "value"},
        )
        result = get_project_model_rpm_limit(user_api_key_dict)
        assert result is None


class TestGetProjectModelTpmLimit:
    """Tests for get_project_model_tpm_limit function."""

    def test_returns_project_metadata_tpm_limit(self):
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            project_metadata={"model_tpm_limit": {"gpt-4": 50000}},
        )
        result = get_project_model_tpm_limit(user_api_key_dict)
        assert result == {"gpt-4": 50000}

    def test_returns_none_when_no_project_metadata(self):
        user_api_key_dict = UserAPIKeyAuth(api_key="sk-123")
        result = get_project_model_tpm_limit(user_api_key_dict)
        assert result is None

    def test_returns_none_when_project_metadata_missing_key(self):
        user_api_key_dict = UserAPIKeyAuth(
            api_key="sk-123",
            project_metadata={"other_key": "value"},
        )
        result = get_project_model_tpm_limit(user_api_key_dict)
        assert result is None


class TestCheckCompleteCredentials:
    """Tests for the api_key validation in check_complete_credentials."""

    def test_returns_false_when_api_key_missing(self):
        result = check_complete_credentials({"model": "gpt-4"})
        assert result is False

    def test_returns_false_when_api_key_is_none(self):
        result = check_complete_credentials({"model": "gpt-4", "api_key": None})
        assert result is False

    def test_returns_false_when_api_key_is_empty_string(self):
        result = check_complete_credentials({"model": "gpt-4", "api_key": ""})
        assert result is False

    def test_returns_false_when_api_key_is_whitespace(self):
        result = check_complete_credentials({"model": "gpt-4", "api_key": "   "})
        assert result is False

    def test_returns_true_when_api_key_is_valid(self):
        result = check_complete_credentials({"model": "gpt-4", "api_key": "sk-valid"})
        assert result is True


class TestIsRequestBodySafeChecksNestedMetadata:
    """Metadata containers are validated with the same banned-param list as the
    request-body root, in every encoding a caller can send them in."""

    @pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
    def test_banned_param_in_metadata_dict_is_rejected(self, metadata_key):
        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    metadata_key: {"api_base": "https://attacker.example"},
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_banned_param_in_json_string_metadata_is_rejected(self):
        with pytest.raises(ValueError, match="aws_sts_endpoint"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "litellm_metadata": json.dumps(
                        {"aws_sts_endpoint": "https://attacker.example"}
                    ),
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_banned_param_in_litellm_params_metadata_is_rejected(self):
        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "litellm_params": {
                        "metadata": {"api_base": "https://attacker.example"}
                    },
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_benign_metadata_is_allowed(self):
        assert (
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "metadata": {"tags": ["prod"], "trace_id": "abc"},
                    "litellm_params": {"metadata": {"tags": ["prod"]}},
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )


class TestIsRequestBodySafeChecksBracketNotationMetadata:
    """Bracket notation is how multipart callers express nested metadata; it is
    validated the same way the dict form is."""

    @pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
    def test_bracket_notation_banned_param_is_rejected(self, metadata_key):
        with pytest.raises(ValueError, match="aws_sts_endpoint"):
            is_request_body_safe(
                request_body={
                    "purpose": "assistants",
                    f"{metadata_key}[aws_sts_endpoint]": "https://example.invalid",
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_bracket_notation_api_base_is_rejected(self):
        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body={"litellm_metadata[api_base]": "https://example.invalid"},
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_bracket_notation_allowed_under_proxy_wide_opt_in(self):
        assert (
            is_request_body_safe(
                request_body={"litellm_metadata[api_base]": "https://byok.example"},
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )

    def test_benign_bracket_notation_metadata_is_allowed(self):
        assert (
            is_request_body_safe(
                request_body={
                    "purpose": "assistants",
                    "litellm_metadata[spend_logs_metadata][owner]": "john",
                    "litellm_metadata[tags]": "production",
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )

    def test_bracket_notation_matches_json_encoding_for_deeper_nesting(self):
        """A value nested below the first level is treated the same either way:
        the check descends one level into metadata, for both encodings."""
        deep_bracket = {
            "litellm_metadata[spend_logs_metadata][api_base]": "https://example.invalid"
        }
        deep_json = {
            "litellm_metadata": {
                "spend_logs_metadata": {"api_base": "https://example.invalid"}
            }
        }
        kwargs = dict(general_settings={}, llm_router=None, model="gpt-4")
        assert is_request_body_safe(request_body=deep_bracket, **kwargs) is True
        assert is_request_body_safe(request_body=deep_json, **kwargs) is True

    def test_body_without_bracket_keys_is_unaffected(self):
        assert (
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )


class TestIsRequestBodySafeBlocksModelList:
    """model_list is an SDK-only field with no proxy API meaning; it must
    be rejected from the request body regardless of any opt-in."""

    def test_model_list_rejected_with_no_opt_in(self):
        with pytest.raises(ValueError, match="model_list is not allowed"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                    "model_list": [
                        {
                            "model_name": "gpt-4",
                            "litellm_params": {
                                "model": "openai/gpt-4",
                                "api_base": "https://attacker.example/v1",
                            },
                        }
                    ],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_model_list_rejected_even_with_proxy_wide_opt_in(self):
        with pytest.raises(ValueError, match="model_list is not allowed"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                    "model_list": [],
                },
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="gpt-4",
            )

    def test_model_list_rejected_even_with_client_api_key(self):
        with pytest.raises(ValueError, match="model_list is not allowed"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "api_key": "sk-caller",
                    "model_list": [],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_normal_body_still_passes(self):
        assert (
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )


class TestIsRequestBodySafeBlocksVertexCredentialAlias:
    @pytest.mark.parametrize("field", ["vertex_ai_credentials"])
    def test_field_in_request_body_is_rejected(self, field):
        with pytest.raises(ValueError, match=field):
            is_request_body_safe(
                request_body={"model": "gpt-4", field: "attacker-supplied"},
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    @pytest.mark.parametrize("field", ["vertex_ai_credentials"])
    def test_admin_opt_in_proxy_wide_allows(self, field):
        assert (
            is_request_body_safe(
                request_body={"model": "gpt-4", field: "byok-supplied"},
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )


def _rounds_deep_api_base_payload(rounds, field):
    """Build a fallbacks payload with ``api_base`` on a target nested ``rounds``
    fallback-rounds deep, each round wrapped in its own grouping dict."""
    node = {"model": "leaf", "api_base": "https://attacker.example"}
    for i in range(rounds):
        node = {"model": f"m{i}", field: [{"grp": [node]}]}
    return {"model": "gpt-4", field: [{"grp": [node]}]}


class TestIsRequestBodySafeBlocksFallbackSmuggle:
    """``is_request_body_safe`` runs the banned-param check on every dict target
    inside the fallback lists."""

    def test_api_base_on_flat_dict_fallback_is_rejected(self):
        """The router merges a dict fallback target into the call kwargs, so an
        ``api_base`` on it would send the deployment's key to that host."""
        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "fallbacks": [
                        {"model": "gpt-4", "api_base": "https://attacker.example"}
                    ],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    @pytest.mark.parametrize(
        "fallback_key",
        ["fallbacks", "context_window_fallbacks", "content_policy_fallbacks"],
    )
    def test_api_base_smuggled_via_nested_fallback_is_rejected(self, fallback_key):
        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    fallback_key: [
                        {
                            "gpt-4": [
                                {"model": "evil", "api_base": "https://attacker.example"},
                            ]
                        }
                    ],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_string_only_fallbacks_are_accepted(self):
        assert (
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "fallbacks": [{"gpt-4": ["gpt-3.5-turbo", "claude-3-haiku"]}],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )

    def test_benign_dict_fallback_entry_is_accepted(self):
        assert (
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "fallbacks": [{"gpt-4": [{"model": "gpt-3.5-turbo"}]}],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )

    def test_smuggled_fallback_allowed_under_proxy_wide_opt_in(self):
        assert (
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "fallbacks": [
                        {"gpt-4": [{"model": "byok", "api_base": "https://my-byok.example"}]}
                    ],
                },
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )

    @pytest.mark.parametrize(
        "fallback_field",
        ["fallbacks", "context_window_fallbacks", "content_policy_fallbacks"],
    )
    @pytest.mark.parametrize("surface", ["top_level", "router_settings_override"])
    def test_deeply_nested_api_base_smuggle_rejected_on_both_surfaces(
        self, fallback_field, surface
    ):
        nested = [
            {
                "always-fail": [
                    {
                        "model": "x",
                        fallback_field: [
                            {"x": [{"model": "deepseek-chat", "api_base": "http://attacker"}]}
                        ],
                    }
                ]
            }
        ]
        request_body = {"model": "gpt-4"}
        if surface == "top_level":
            request_body[fallback_field] = nested
        else:
            request_body["router_settings_override"] = {fallback_field: nested}
        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body=request_body,
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_router_settings_override_single_level_api_base_rejected(self):
        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "router_settings_override": {
                        "fallbacks": [{"gpt-4": [{"model": "x", "api_base": "http://attacker"}]}]
                    },
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_model_less_config_dict_api_base_rejected(self):
        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "fallbacks": [{"gpt-4": [{"api_base": "http://attacker"}]}],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_nested_api_base_caught_across_router_fallback_rounds(self):
        """An ``api_base`` target nested ``ROUTER_MAX_FALLBACKS - 1`` rounds deep
        is still reached and rejected."""
        import litellm

        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body=_rounds_deep_api_base_payload(
                    litellm.ROUTER_MAX_FALLBACKS - 1, "fallbacks"
                ),
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_grouping_only_deep_chain_is_rejected_at_depth_limit(self):
        """A deep grouping-only chain (``{"g": [{"g": [...]}]}``) is rejected at the
        validation-depth limit rather than accepted or raising RecursionError."""
        node: object = ["safe-model"]
        for _ in range(5000):
            node = [{"grp": node}]
        with pytest.raises(ValueError, match="depth"):
            is_request_body_safe(
                request_body={"model": "gpt-4", "fallbacks": node},
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_pathologically_deep_model_nesting_is_rejected(self):
        with pytest.raises(ValueError, match="depth"):
            is_request_body_safe(
                request_body=_rounds_deep_api_base_payload(5000, "fallbacks"),
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )


class TestIsRequestBodySafeRejectsUrlValuedFallback:
    @pytest.mark.parametrize(
        "fallback_field",
        ["fallbacks", "context_window_fallbacks", "content_policy_fallbacks"],
    )
    def test_url_valued_string_fallback_is_rejected(self, fallback_field):
        with pytest.raises(ValueError, match="URL-valued fallback"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    fallback_field: [
                        {"gpt-4": ["huggingface/http://attacker.example/path"]}
                    ],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    @pytest.mark.parametrize(
        "fallback_field",
        ["fallbacks", "context_window_fallbacks", "content_policy_fallbacks"],
    )
    def test_url_valued_dict_model_fallback_is_rejected(self, fallback_field):
        with pytest.raises(ValueError, match="URL-valued fallback"):
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    fallback_field: [
                        {"gpt-4": [{"model": "huggingface/http://attacker.example/path"}]}
                    ],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )

    def test_ordinary_string_fallback_is_allowed(self):
        assert (
            is_request_body_safe(
                request_body={"model": "gpt-4", "fallbacks": [{"gpt-4": ["gpt-4-backup"]}]},
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )

    def test_ordinary_dict_model_fallback_is_allowed(self):
        assert (
            is_request_body_safe(
                request_body={
                    "model": "gpt-4",
                    "fallbacks": [{"gpt-4": [{"model": "gpt-4-backup"}]}],
                },
                general_settings={},
                llm_router=None,
                model="gpt-4",
            )
            is True
        )
