from unittest.mock import MagicMock, call, patch

import pytest

from dlms_cosem import exceptions
from dlms_cosem.client import DlmsClient
from dlms_cosem.hdlc import address, connection, frames, state
from dlms_cosem.hdlc.exceptions import HdlcParsingError
from dlms_cosem.io import BlockingTcpIO, HdlcTransport, IPTransport, LLC_RESPONSE_HEADER
from dlms_cosem.security import NoSecurityAuthentication


class DummyIo:
    def __init__(self):
        self.sent_data = []
        self.flush_called = False

    def connect(self) -> None:
        return

    def disconnect(self) -> None:
        return

    def send(self, data: bytes) -> None:
        self.sent_data.append(data)

    def recv(self, amount: int = 1) -> bytes:
        return b""

    def recv_until(self, end: bytes) -> bytes:
        return b""

    def flush_input(self) -> None:
        self.flush_called = True


def test_hdlc_transport_defaults():
    transport = HdlcTransport(
        client_logical_address=16,
        server_logical_address=1,
        io=DummyIo(),
    )
    assert transport.retries == 3
    assert transport.retry_delay == 0.5


def test_send_request_succeeds_without_retries():
    io = DummyIo()
    transport = HdlcTransport(
        client_logical_address=16,
        server_logical_address=1,
        io=io,
        retries=2,
        retry_delay=0.1,
    )

    mock_response = MagicMock()
    mock_response.payload = LLC_RESPONSE_HEADER + b"test_payload"
    mock_response.segmented = False
    mock_response.final = True

    with patch.object(transport, "next_event", return_value=mock_response):
        result = transport.send_request(b"test_telegram")
        assert result == b"test_payload"


def test_send_request_retries_on_timeout_and_succeeds():
    io = DummyIo()
    transport = HdlcTransport(
        client_logical_address=16,
        server_logical_address=1,
        io=io,
        retries=2,
        retry_delay=0.5,
    )

    mock_response = MagicMock()
    mock_response.payload = LLC_RESPONSE_HEADER + b"success_data"
    mock_response.segmented = False
    mock_response.final = True

    # Attempt 1 raises CommunicationTimeoutError, attempt 2 returns response
    with (
        patch.object(
            transport,
            "next_event",
            side_effect=[
                exceptions.CommunicationTimeoutError("Timed out"),
                mock_response,
            ],
        ),
        patch("time.sleep") as mock_sleep,
    ):
        result = transport.send_request(b"request_bytes")
        assert result == b"success_data"
        assert io.flush_called is True
        mock_sleep.assert_called_once_with(0.5)


def test_send_request_restores_sequence_numbers_on_retry():
    io = DummyIo()
    transport = HdlcTransport(
        client_logical_address=16,
        server_logical_address=1,
        io=io,
        retries=2,
        retry_delay=0.0,
    )
    transport.hdlc_connection.state.current_state = state.IDLE
    transport.hdlc_connection.server_ssn = 3
    transport.hdlc_connection.server_rsn = 2
    transport.hdlc_connection.client_ssn = 1
    transport.hdlc_connection.client_rsn = 4

    captured_ssn = []

    original_send_frame = transport.send_frame

    def mock_send_frame(frame):
        captured_ssn.append(frame.send_sequence_number)
        original_send_frame(frame)

    mock_response = MagicMock()
    mock_response.payload = LLC_RESPONSE_HEADER + b"ok"
    mock_response.segmented = False
    mock_response.final = True

    with (
        patch.object(transport, "send_frame", side_effect=mock_send_frame),
        patch.object(
            transport,
            "next_event",
            side_effect=[
                exceptions.CommunicationError("Noise error"),
                mock_response,
            ],
        ),
    ):
        result = transport.send_request(b"test")
        assert result == b"ok"
        # Both transmission attempts used the exact same initial sequence number
        assert captured_ssn == [3, 3]


def test_send_request_exhausts_retries_and_raises():
    io = DummyIo()
    transport = HdlcTransport(
        client_logical_address=16,
        server_logical_address=1,
        io=io,
        retries=2,
        retry_delay=0.1,
    )

    with (
        patch.object(
            transport,
            "next_event",
            side_effect=exceptions.CommunicationTimeoutError("Timed out"),
        ),
        patch("time.sleep") as mock_sleep,
    ):
        with pytest.raises(exceptions.CommunicationTimeoutError):
            transport.send_request(b"request")

        # 2 retries -> 2 sleeps
        assert mock_sleep.call_count == 2


def test_send_request_retries_on_hdlc_parsing_error():
    io = DummyIo()
    transport = HdlcTransport(
        client_logical_address=16,
        server_logical_address=1,
        io=io,
        retries=2,
        retry_delay=0.0,
    )

    mock_response = MagicMock()
    mock_response.payload = LLC_RESPONSE_HEADER + b"recovered"
    mock_response.segmented = False
    mock_response.final = True

    with patch.object(
        transport,
        "next_event",
        side_effect=[
            HdlcParsingError("FCS is not correct"),
            mock_response,
        ],
    ):
        result = transport.send_request(b"data")
        assert result == b"recovered"


def test_send_request_retries_on_value_error_missing_llc_header():
    io = DummyIo()
    transport = HdlcTransport(
        client_logical_address=16,
        server_logical_address=1,
        io=io,
        retries=2,
        retry_delay=0.0,
    )

    mock_bad = MagicMock()
    mock_bad.payload = b"bad_payload_without_llc"
    mock_bad.segmented = False
    mock_bad.final = True

    mock_good = MagicMock()
    mock_good.payload = LLC_RESPONSE_HEADER + b"good_payload"
    mock_good.segmented = False
    mock_good.final = True

    with patch.object(transport, "next_event", side_effect=[mock_bad, mock_good]):
        result = transport.send_request(b"data")
        assert result == b"good_payload"


def test_connect_retries_snrm_and_succeeds():
    io = DummyIo()
    transport = HdlcTransport(
        client_logical_address=16,
        server_logical_address=1,
        io=io,
        retries=2,
        retry_delay=0.5,
    )

    mock_ua = MagicMock(spec=frames.UnNumberedAcknowledgmentFrame)

    with (
        patch.object(
            transport,
            "next_event",
            side_effect=[
                exceptions.CommunicationTimeoutError("No UA response"),
                mock_ua,
            ],
        ),
        patch("time.sleep") as mock_sleep,
    ):
        res = transport.connect()
        assert res == mock_ua
        mock_sleep.assert_called_once_with(0.5)


def test_connect_exhausts_retries_and_raises():
    io = DummyIo()
    transport = HdlcTransport(
        client_logical_address=16,
        server_logical_address=1,
        io=io,
        retries=1,
        retry_delay=0.1,
    )

    with (
        patch.object(
            transport,
            "next_event",
            side_effect=exceptions.CommunicationTimeoutError("No UA response"),
        ),
        patch("time.sleep"),
    ):
        with pytest.raises(exceptions.CommunicationTimeoutError):
            transport.connect()


def test_ip_transport_retries_and_succeeds():
    io = DummyIo()
    transport = IPTransport(
        client_logical_address=1,
        server_logical_address=1,
        io=io,
        retries=2,
        retry_delay=0.5,
    )

    with (
        patch.object(
            transport,
            "recv_response",
            side_effect=[
                exceptions.CommunicationError("Socket dropped"),
                b"response_bytes",
            ],
        ),
        patch("time.sleep") as mock_sleep,
    ):
        res = transport.send_request(b"apdu")
        assert res == b"response_bytes"
        mock_sleep.assert_called_once_with(0.5)


def test_ip_transport_exhausts_retries_raises():
    io = DummyIo()
    transport = IPTransport(
        client_logical_address=1,
        server_logical_address=1,
        io=io,
        retries=1,
        retry_delay=0.1,
    )

    with (
        patch.object(
            transport,
            "recv_response",
            side_effect=exceptions.CommunicationError("Socket failed"),
        ),
        patch("time.sleep"),
    ):
        with pytest.raises(exceptions.CommunicationError):
            transport.send_request(b"apdu")


def test_dlms_client_forwards_retries_and_delay_to_transport():
    transport = HdlcTransport(
        client_logical_address=16,
        server_logical_address=1,
        io=DummyIo(),
    )
    client = DlmsClient(
        transport=transport,
        authentication=NoSecurityAuthentication(),
        retries=5,
        retry_delay=1.2,
    )

    assert client.transport.retries == 5
    assert client.transport.retry_delay == 1.2
