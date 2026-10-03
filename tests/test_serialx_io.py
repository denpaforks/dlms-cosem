import socket
import threading
import time
import pytest

from dlms_cosem import exceptions
from dlms_cosem.hdlc import frames
from dlms_cosem.io import IoImplementation, SerialXIO, HdlcTransport


@pytest.fixture
def tcp_server():
    """Fixture providing a simple echo/responder TCP server for socket:// testing."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    stop_event = threading.Event()
    connections = []

    def handle():
        while not stop_event.is_set():
            try:
                server.settimeout(0.2)
                conn, _ = server.accept()
                connections.append(conn)
            except (socket.timeout, OSError):
                continue

            try:
                while not stop_event.is_set():
                    conn.settimeout(0.2)
                    try:
                        data = conn.recv(1024)
                        if not data:
                            break
                        conn.sendall(data)
                    except (socket.timeout, OSError):
                        continue
            except Exception:
                pass
            finally:
                conn.close()

    thread = threading.Thread(target=handle, daemon=True)
    thread.start()

    yield f"socket://127.0.0.1:{port}"

    stop_event.set()
    for conn in connections:
        try:
            conn.close()
        except Exception:
            pass
    server.close()
    thread.join(timeout=1)


class TestSerialXIO:

    def test_satisfies_io_protocol(self):
        io = SerialXIO(port_url="socket://127.0.0.1:10000")
        for method in ("connect", "disconnect", "send", "recv", "recv_until"):
            assert hasattr(io, method)
            assert callable(getattr(io, method))

    def test_missing_serialx_raises_import_error(self, monkeypatch):
        import dlms_cosem.io as dlms_io
        monkeypatch.setattr(dlms_io, "serialx", None)

        with pytest.raises(ImportError, match="serialx is required"):
            SerialXIO(port_url="socket://127.0.0.1:10000")

    def test_connect_and_disconnect(self, tcp_server):
        io = SerialXIO(port_url=tcp_server, timeout=2)
        assert io.serial_port is None

        io.connect()
        assert io.serial_port is not None
        assert io.serial_port.is_open

        io.disconnect()
        assert io.serial_port is None

    def test_connect_when_already_connected_raises(self, tcp_server):
        io = SerialXIO(port_url=tcp_server, timeout=2)
        io.connect()

        with pytest.raises(RuntimeError, match="already is open"):
            io.connect()

        io.disconnect()

    def test_operations_on_closed_port_raise(self):
        io = SerialXIO(port_url="socket://127.0.0.1:10000")

        with pytest.raises(RuntimeError, match="closed serial port"):
            io.send(b"test")

        with pytest.raises(RuntimeError, match="closed serial port"):
            io.recv(1)

        with pytest.raises(RuntimeError, match="closed serial port"):
            io.recv_until(b"~")

    def test_send_and_recv(self, tcp_server):
        io = SerialXIO(port_url=tcp_server, timeout=2)
        io.connect()

        payload = b"hello dlms"
        io.send(payload)
        received = io.recv(len(payload))

        assert received == payload
        io.disconnect()

    def test_recv_until(self, tcp_server):
        io = SerialXIO(port_url=tcp_server, timeout=2)
        io.connect()

        payload = b"frame_start~"
        io.send(payload)
        received = io.recv_until(b"~")

        assert received == payload
        io.disconnect()

    def test_connection_failure_raises_communication_error(self):
        # Port 1 is typically unused and closed, connection should fail
        io = SerialXIO(port_url="socket://127.0.0.1:1", timeout=1)

        with pytest.raises(exceptions.CommunicationError, match="Unable to open serial port"):
            io.connect()

    def test_hdlc_transport_integration(self):
        io = SerialXIO(port_url="socket://127.0.0.1:10000")
        transport = HdlcTransport(
            client_logical_address=16,
            server_logical_address=1,
            io=io,
        )
        assert transport.io is io
