import os
import queue
import socket
import threading
import time
import warnings
from enum import IntEnum
import winsound

import customtkinter as ctk
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from scapy.all import IP, TCP, sniff

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class AppConfig:
    TITLE = "DetectionGuard"
    INITIAL_WIDTH = 1100
    INITIAL_HEIGHT = 700
    MIN_WIDTH = 900
    MIN_HEIGHT = 600


class Assets:
    ICON_PATH = "../ui_assets/icon.ico"
    LOGO_PATH = "../ui_assets/detectionguard_logo.png"
    MODEL_PATH = "../assets/DetectionGuard_model.h5"
    SCALER_PATH = "../assets/scaler.gz"
    ENCODER_PATH = "../assets/encoder.gz"


class NetworkConfig:
    LIVE_INTERFACE_NAME = "Intel(R) Wi-Fi 6 AX203"
    SIMULATION_BIND_IP = "0.0.0.0"
    SIMULATION_PORT = 9999
    SOCKET_BUFFER_SIZE = 4096
    SOCKET_LISTEN_BACKLOG = 5
    SOCKET_TIMEOUT_SECONDS = 1.0
    THREAD_JOIN_TIMEOUT_SECONDS = 1.5


class ModelConfig:
    UI_ALERT_CONFIDENCE_THRESHOLD = 60.0
    LOG_ALERT_CONFIDENCE_THRESHOLD = 30.0
    NORMAL_TRAFFIC_LABEL = "Normal Traffic"
    FEATURE_VECTOR_SIZE = 52


class FeatureIndex(IntEnum):
    DESTINATION_PORT = 0
    FORWARD_PACKET_LENGTH_MAX = 4
    FORWARD_PACKET_LENGTH_MIN = 5
    FORWARD_PACKET_LENGTH_MEAN = 6
    FIN_FLAG_COUNT = 37
    PSH_FLAG_COUNT = 38
    ACK_FLAG_COUNT = 39
    INIT_WIN_BYTES_FORWARD = 42


class ColorPalette:
    PANEL_BACKGROUND = "#2B2B2B"
    STATUS_GREEN = "#2CC985"
    STATUS_GREEN_DARK = "#26a271"
    STATUS_RED = "#E64444"
    STATUS_BLUE = "#3A7EBF"
    TEXT_GRAY = "#9A9A9A"


class FontConfig:
    TITLE = ("Arial", 20, "bold")
    BUTTON_LARGE = ("Arial", 18, "bold")
    METRIC_LABEL = ("Arial", 12)
    METRIC_VALUE = ("Arial", 28, "bold")
    STATUS = ("Arial", 16)
    LOG_HEADER = ("Arial", 14, "bold")
    LOG_TEXT = ("Courier New", 12)


class TextContent:
    SCAN_START = "Start Scanning"; SCAN_STOP = "Stop Scanning"
    STATUS_IDLE = "Status: System Idle"
    STATUS_MONITORING = "Status: Monitoring Active ({mode} Mode)..."
    STATUS_MODE_SWITCH = "Status: Switched to {mode} Mode. Ready."
    PACKETS_SCANNED = "Packets Scanned"; THREATS_DETECTED = "Threats Detected"
    LIVE_ACTIVITY = "Live Activity Log"
    ALERT_BANNER = "ALERT: {label} DETECTED (Confidence: {confidence:.1f}%)"
    LOG_ALERT_DETAIL = "{alert} | Source: {source}"
    LOG_INFO_STARTED = "INFO: {mode} Mode started..."
    LOG_INFO_STOPPED = "INFO: Scanning stopped."; LOG_INFO_CLEARED = "INFO: Logs cleared."
    THEME_LABEL = "Theme:"; THEME_DARK = "Dark"; CLEAR_BUTTON = "Clear"


class FeatureExtractor:
    @staticmethod
    def from_packet(packet) -> np.ndarray:
        features = np.zeros(ModelConfig.FEATURE_VECTOR_SIZE, dtype=float)
        if packet.haslayer(IP) and packet.haslayer(TCP):
            tcp_layer, ip_layer = packet[TCP], packet[IP]
            payload_len = float(len(tcp_layer.payload))
            flags = str(tcp_layer.flags)
            features[FeatureIndex.DESTINATION_PORT] = tcp_layer.dport
            features[FeatureIndex.FORWARD_PACKET_LENGTH_MAX] = payload_len
            features[FeatureIndex.FORWARD_PACKET_LENGTH_MIN] = payload_len
            features[FeatureIndex.FORWARD_PACKET_LENGTH_MEAN] = payload_len
            features[FeatureIndex.INIT_WIN_BYTES_FORWARD] = ip_layer.window
            if "F" in flags: features[FeatureIndex.FIN_FLAG_COUNT] = 1.0
            if "P" in flags: features[FeatureIndex.PSH_FLAG_COUNT] = 1.0
            if "A" in flags: features[FeatureIndex.ACK_FLAG_COUNT] = 1.0
        return features


class ModelPredictor:
    def __init__(self, model, scaler, encoder):
        self.model, self.scaler, self.encoder = model, scaler, encoder
        self.input_shape = self.model.input_shape[1]

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        scaled = self.scaler.transform([features])
        reshaped = scaled.reshape(1, self.input_shape)
        probabilities = self.model.predict(reshaped, verbose=0)[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index] * 100.0)
        label = self.encoder.inverse_transform([index])[0]
        return label, confidence


class PacketListener:
    def __init__(self, packet_queue: queue.Queue, stop_event: threading.Event):
        self.packet_queue, self.stop_event = packet_queue, stop_event
        self.listener_thread = None

    def start(self): raise NotImplementedError
    def stop(self):
        self.stop_event.set()
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=NetworkConfig.THREAD_JOIN_TIMEOUT_SECONDS)


class LivePacketSniffer(PacketListener):
    def __init__(self, packet_queue: queue.Queue, stop_event: threading.Event, interface: str):
        super().__init__(packet_queue, stop_event)
        self.interface = interface

    def _sniff_loop(self):
        sniff(iface=self.interface, prn=self.packet_queue.put, stop_filter=lambda p: self.stop_event.is_set())

    def start(self):
        self.listener_thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.listener_thread.start()


class SimulationSocketListener(PacketListener):
    def __init__(self, packet_queue: queue.Queue, stop_event: threading.Event, bind_ip: str, port: int):
        super().__init__(packet_queue, stop_event)
        self.bind_ip, self.port = bind_ip, port

    def _listen_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.bind_ip, self.port))
        server.listen(NetworkConfig.SOCKET_LISTEN_BACKLOG)
        server.settimeout(NetworkConfig.SOCKET_TIMEOUT_SECONDS)
        while not self.stop_event.is_set():
            try:
                conn, addr = server.accept()
                with conn:
                    data = conn.recv(NetworkConfig.SOCKET_BUFFER_SIZE).decode()
                    if data:
                        features = np.fromstring(data, sep=",")
                        packet_data = {"features": features, "source": addr[0]}
                        self.packet_queue.put(packet_data)
            except socket.timeout:
                continue
        server.close()

    def start(self):
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()


class DetectionGuardApp(ctk.CTk):
    def __init__(self, model, scaler, encoder):
        super().__init__()
        self.predictor = ModelPredictor(model, scaler, encoder)
        self.is_scanning = False; self.packets_scanned = 0; self.threats_detected = 0
        self.packet_queue = queue.Queue(); self.stop_event = threading.Event()
        self.active_listener = None; self.processor_thread = None
        self._configure_window()
        self._build_ui()

    def _configure_window(self):
        self.title(AppConfig.TITLE)
        self.geometry(f"{AppConfig.INITIAL_WIDTH}x{AppConfig.INITIAL_HEIGHT}")
        self.minsize(AppConfig.MIN_WIDTH, AppConfig.MIN_HEIGHT)
        try: self.iconbitmap(Assets.ICON_PATH)
        except Exception: print(f"Icon not found at: {Assets.ICON_PATH}")

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        self._create_sidebar()
        self._create_main_panel()

    def _create_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew"); sidebar.grid_rowconfigure(5, weight=1)
        try:
            logo = ctk.CTkImage(Image.open(Assets.LOGO_PATH), size=(100, 100))
            ctk.CTkLabel(sidebar, image=logo, text="").grid(row=0, column=0, padx=20, pady=(20, 10))
        except Exception:
            ctk.CTkLabel(sidebar, text=AppConfig.TITLE, font=FontConfig.TITLE).grid(row=0, column=0, padx=20, pady=(20, 10))
        ctk.CTkLabel(sidebar, text=TextContent.THEME_LABEL, anchor="w").grid(row=6, column=0, padx=20, pady=(10, 0), sticky="w")
        self.theme_switch = ctk.CTkSwitch(sidebar, text=TextContent.THEME_DARK, command=self._handle_theme_toggle)
        self.theme_switch.grid(row=7, column=0, padx=20, pady=(0, 20), sticky="w"); self.theme_switch.select()

    def _create_main_panel(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        main.grid_columnconfigure(0, weight=1); main.grid_rowconfigure(3, weight=1)
        
        top_bar = ctk.CTkFrame(main, fg_color="transparent")
        top_bar.grid(row=0, column=0, sticky="ew"); top_bar.grid_columnconfigure((1, 2), weight=1)
        self.scan_button = ctk.CTkButton(top_bar, text=TextContent.SCAN_START, command=self._toggle_scanning, height=50, width=250, font=FontConfig.BUTTON_LARGE)
        self.scan_button.grid(row=0, column=0, padx=(0, 20), pady=10)
        self.packets_label = self._create_metric_card(top_bar, TextContent.PACKETS_SCANNED, ColorPalette.STATUS_BLUE, 1)
        self.threats_label = self._create_metric_card(top_bar, TextContent.THREATS_DETECTED, ColorPalette.STATUS_RED, 2)
        
        self.mode_selection = ctk.CTkSegmentedButton(main, values=["Live Network", "Simulation"], command=self._handle_mode_change)
        self.mode_selection.set("Live Network"); self.mode_selection.grid(row=1, column=0, pady=10, sticky="ew")
        
        status_frame = ctk.CTkFrame(main, fg_color=ColorPalette.PANEL_BACKGROUND)
        status_frame.grid(row=2, column=0, pady=10, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        
        self.health_status_indicator = ctk.CTkFrame(status_frame, width=20, height=20, corner_radius=10, fg_color=ColorPalette.TEXT_GRAY)
        self.health_status_indicator.grid(row=0, column=0, padx=(20, 10), pady=10)
        
        self.status_label = ctk.CTkLabel(status_frame, text=TextContent.STATUS_IDLE, font=FontConfig.STATUS, text_color=ColorPalette.TEXT_GRAY)
        self.status_label.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        log_panel = ctk.CTkFrame(main, fg_color="transparent")
        log_panel.grid(row=3, column=0, sticky="nsew"); log_panel.grid_columnconfigure(0, weight=1); log_panel.grid_rowconfigure(1, weight=1)
        log_header = ctk.CTkFrame(log_panel, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew"); log_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(log_header, text=TextContent.LIVE_ACTIVITY, font=FontConfig.LOG_HEADER).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(log_header, text=TextContent.CLEAR_BUTTON, width=60, command=self._clear_logs).grid(row=0, column=1, sticky="e")
        self.log_textbox = ctk.CTkTextbox(log_panel, font=FontConfig.LOG_TEXT, state="disabled", wrap="word")
        self.log_textbox.grid(row=1, column=0, sticky="nsew", pady=(5, 0))

    def _create_metric_card(self, parent, label_text, value_color, grid_column):
        card = ctk.CTkFrame(parent, fg_color=ColorPalette.PANEL_BACKGROUND)
        card.grid(row=0, column=grid_column, padx=5, sticky="ew")
        ctk.CTkLabel(card, text=label_text, font=FontConfig.METRIC_LABEL, text_color=ColorPalette.TEXT_GRAY).pack(padx=10, pady=(10, 0))
        value_label = ctk.CTkLabel(card, text="0", font=FontConfig.METRIC_VALUE, text_color=value_color)
        value_label.pack(padx=10, pady=(0, 10))
        return value_label

    def _toggle_scanning(self):
        if self.is_scanning: self._stop_scanning()
        else: self._start_scanning()

    def _start_scanning(self):
        self.is_scanning = True
        self.stop_event.clear()
        self.packets_scanned, self.threats_detected = 0, 0
        self.packets_label.configure(text="0"); self.threats_label.configure(text="0")
        
        self.scan_button.configure(text=TextContent.SCAN_STOP, fg_color=ColorPalette.STATUS_GREEN, hover_color=ColorPalette.STATUS_RED)
        mode = self.mode_selection.get()
        self.status_label.configure(text=TextContent.STATUS_MONITORING.format(mode=mode), text_color=ColorPalette.STATUS_GREEN)
        self.health_status_indicator.configure(fg_color=ColorPalette.STATUS_GREEN)
        self._log_message(TextContent.LOG_INFO_STARTED.format(mode=mode))
        
        if mode == "Live Network":
            self.active_listener = LivePacketSniffer(self.packet_queue, self.stop_event, NetworkConfig.LIVE_INTERFACE_NAME)
        else:
            self.active_listener = SimulationSocketListener(self.packet_queue, self.stop_event, NetworkConfig.SIMULATION_BIND_IP, NetworkConfig.SIMULATION_PORT)
        
        self.active_listener.start()
        self.processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processor_thread.start()

    def _stop_scanning(self):
        self.is_scanning = False
        if self.active_listener: self.active_listener.stop()
        self.stop_event.set()
        
        default_fg = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
        default_hover = ctk.ThemeManager.theme["CTkButton"]["hover_color"]
        self.scan_button.configure(text=TextContent.SCAN_START, fg_color=default_fg, hover_color=default_hover)
        self.status_label.configure(text=TextContent.STATUS_IDLE, text_color=ColorPalette.TEXT_GRAY)
        self.health_status_indicator.configure(fg_color=ColorPalette.TEXT_GRAY)
        self._log_message(TextContent.LOG_INFO_STOPPED)

    def _processing_loop(self):
        while not self.stop_event.is_set():
            try:
                item = self.packet_queue.get(timeout=1.0)
                source_info, features = self._normalize_packet_data(item)
                if features is not None and np.any(features):
                    self._process_features(features, source_info)
            except queue.Empty: continue

    def _normalize_packet_data(self, data):
        if isinstance(data, dict):
            return data.get("source", "Simulation"), data.get("features")
        else:
            source = data[IP].src if data.haslayer(IP) else "N/A"
            return source, FeatureExtractor.from_packet(data)

    def _process_features(self, features: np.ndarray, source_info: str):
        self.packets_scanned += 1
        self.packets_label.configure(text=f"{self.packets_scanned:,}")
        
        label, confidence = self.predictor.predict(features)
        
        is_loggable_threat = label != ModelConfig.NORMAL_TRAFFIC_LABEL and confidence > ModelConfig.LOG_ALERT_CONFIDENCE_THRESHOLD
        if is_loggable_threat:
            is_high_confidence_alert = confidence > ModelConfig.UI_ALERT_CONFIDENCE_THRESHOLD
            self.after(0, self._handle_threat_detection, label, confidence, source_info, is_high_confidence_alert)
        else:
            self.after(0, self._handle_normal_traffic)

    def _handle_threat_detection(self, label: str, confidence: float, source: str, is_high_confidence: bool):
        self.threats_detected += 1
        self.threats_label.configure(text=str(self.threats_detected))
        
        alert = TextContent.ALERT_BANNER.format(label=label.upper(), confidence=confidence)
        log_details = TextContent.LOG_ALERT_DETAIL.format(alert=alert, source=source)
        self._log_message(log_details)

        if is_high_confidence:
            winsound.Beep(800, 250)
            self.health_status_indicator.configure(fg_color=ColorPalette.STATUS_RED)
            self.status_label.configure(text=alert, text_color=ColorPalette.STATUS_RED)

    def _handle_normal_traffic(self):
        if self.health_status_indicator.cget("fg_color") != ColorPalette.STATUS_GREEN:
            self.health_status_indicator.configure(fg_color=ColorPalette.STATUS_GREEN)
            mode = self.mode_selection.get()
            self.status_label.configure(text=TextContent.STATUS_MONITORING.format(mode=mode), text_color=ColorPalette.STATUS_GREEN)

    def _log_message(self, message: str):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("0.0", f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_textbox.configure(state="disabled")

    def _handle_mode_change(self, mode: str):
        if self.is_scanning: self._stop_scanning()
        self.status_label.configure(text=TextContent.STATUS_MODE_SWITCH.format(mode=mode), text_color=ColorPalette.TEXT_GRAY)

    def _handle_theme_toggle(self):
        ctk.set_appearance_mode("Dark" if self.theme_switch.get() else "Light")

    def _clear_logs(self):
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")
        self._log_message(TextContent.LOG_INFO_CLEARED)


def main():
    print("[INFO] Loading ML model and assets...")
    model = tf.keras.models.load_model(Assets.MODEL_PATH)
    scaler = joblib.load(Assets.SCALER_PATH)
    encoder = joblib.load(Assets.ENCODER_PATH)
    print("[SUCCESS] Assets loaded.")
    app = DetectionGuardApp(model=model, scaler=scaler, encoder=encoder)
    app.mainloop()


if __name__ == "__main__":
    main()