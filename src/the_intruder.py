import customtkinter as ctk
import threading
import pandas as pd
from scapy.all import IP, TCP, Raw, send
import random
import sys
import glob
import os
import time
import socket
from pathlib import Path
class AppConfig:
    TITLE = "DetectionGuard - Intruder"
    GEOMETRY = "500x400"

class AssetConfig:
    PLAYBOOK_DIR = Path("../assets")
    PLAYBOOK_PATTERN = "TP_*_Attacks.csv"

class NetworkConfig:
    SIMULATION_PORT = 9999

class AttackConfig:
    BARRAGE_PACKET_COUNT = 50
    SINGLE_ATTACK_PACKET_COUNT = 200
    BARRAGE_MODE_LABEL = "-- BARRAGE --"

class UIContent:
    TITLE = "Intruder Control Panel"
    IP_PLACEHOLDER = "Enter Victim IP Address"
    LAUNCH_BUTTON = "Launch Attack"
    LAUNCH_IN_PROGRESS = "Attack in Progress..."
    NO_PLAYBOOKS = "No Playbooks Found"
    STATUS_READY = "Status: Ready"
    ERROR_NO_IP = "Status: Error - Please enter a target IP address."
    ERROR_NO_PLAYBOOKS = "Status: Error - No attack playbooks found."
    STATUS_BARRAGE_START = "Status: Starting Attack Barrage ({mode} Mode)..."
    STATUS_BARRAGE_COMPLETE = "Status: Attack Barrage complete!"
    STATUS_SINGLE_START = "Status: Launching '{type}' via {mode}..."
    STATUS_SINGLE_COMPLETE = "Status: Attack on '{type}' complete!"
    STATUS_ERROR = "Status: Error - {error}"

class PacketFactory:
    @staticmethod
    def create_scapy_packet(target_ip, vector):
        dst_port = int(vector.get('Destination Port', 80))
        window_size = int(vector.get('Init_Win_bytes_forward', 29200))
        payload_size = max(0, int(vector.get('Fwd Packet Length Max', 50) - 40))
        payload = "X" * payload_size
        flag = random.choice(["S", "F", "UAP", "SRA"])
        return IP(dst=target_ip)/TCP(dport=dst_port, window=window_size, flags=flag)/Raw(load=payload)

    @staticmethod
    def send_vector_via_socket(target_ip, vector):
        feature_vector = vector.drop('Attack Type').values
        vector_string = ",".join(map(str, feature_vector))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((target_ip, NetworkConfig.SIMULATION_PORT))
            s.sendall(vector_string.encode())

class PlaybookManager:
    def __init__(self, directory, pattern):
        self.directory = directory
        self.pattern = pattern

    def find_available_attacks(self):
        playbook_paths = list(self.directory.glob(self.pattern))
        attack_names = [p.stem.replace("TP_", "").replace("_Attacks", "").replace("_", " ") for p in playbook_paths]
        if not attack_names:
            return [UIContent.NO_PLAYBOOKS]
        return [AttackConfig.BARRAGE_MODE_LABEL] + sorted(attack_names)

    def load_playbook(self, attack_type):
        safe_name = attack_type.replace(' ', '_').replace('/', '_')
        filename = self.directory / f"TP_{safe_name}_Attacks.csv"
        return pd.read_csv(filename)

class AttackRunner:
    def __init__(self, ui_updater, playbook_manager):
        self.update_ui_status = ui_updater
        self.playbook_manager = playbook_manager

    def execute(self, target_ip, selected_attack, attack_method, all_attack_names):
        try:
            if selected_attack == AttackConfig.BARRAGE_MODE_LABEL:
                self._run_barrage_attack(target_ip, attack_method, all_attack_names)
            else:
                self._run_single_attack(target_ip, selected_attack, attack_method, AttackConfig.SINGLE_ATTACK_PACKET_COUNT)
                self.update_ui_status(UIContent.STATUS_SINGLE_COMPLETE.format(type=selected_attack), "green")
        except Exception as e:
            self.update_ui_status(UIContent.STATUS_ERROR.format(error=e), "red")

    def _run_barrage_attack(self, target_ip, attack_method, all_attack_names):
        self.update_ui_status(UIContent.STATUS_BARRAGE_START.format(mode=attack_method), "gray")
        for attack_type in all_attack_names[1:]:
            self._run_single_attack(target_ip, attack_type, attack_method, AttackConfig.BARRAGE_PACKET_COUNT)
            time.sleep(1)
        self.update_ui_status(UIContent.STATUS_BARRAGE_COMPLETE, "green")

    def _run_single_attack(self, target_ip, attack_type, attack_method, num_packets):
        self.update_ui_status(UIContent.STATUS_SINGLE_START.format(type=attack_type, mode=attack_method), "gray")
        playbook = self.playbook_manager.load_playbook(attack_type)
        
        for _ in range(num_packets):
            random_vector = playbook.sample(1).iloc[0]
            if attack_method == "Live Network":
                packet = PacketFactory.create_scapy_packet(target_ip, random_vector)
                send(packet, verbose=0)
            else:
                PacketFactory.send_vector_via_socket(target_ip, random_vector)
        
        print(f"Finished sending {num_packets} packets for {attack_type}")

class IntruderApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(AppConfig.TITLE)
        self.geometry(AppConfig.GEOMETRY)
        self.grid_columnconfigure(0, weight=1)
        
        self.playbook_manager = PlaybookManager(AssetConfig.PLAYBOOK_DIR, AssetConfig.PLAYBOOK_PATTERN)
        self.attack_runner = AttackRunner(self._update_status_label, self.playbook_manager)
        
        self._create_widgets()

    def _create_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(main_frame, text=UIContent.TITLE, font=("Arial", 20, "bold")).grid(row=0, column=0, padx=10, pady=(10, 20))

        self.ip_entry = ctk.CTkEntry(main_frame, placeholder_text=UIContent.IP_PLACEHOLDER, height=35, font=("Arial", 14))
        self.ip_entry.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.attack_names = self.playbook_manager.find_available_attacks()
        self.attack_menu = ctk.CTkOptionMenu(main_frame, values=self.attack_names, height=35, font=("Arial", 14))
        self.attack_menu.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.launch_button = ctk.CTkButton(main_frame, text=UIContent.LAUNCH_BUTTON, command=self._handle_launch_click, height=40, font=("Arial", 16))
        self.launch_button.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.attack_method_selection = ctk.CTkSegmentedButton(main_frame, values=["Live Network", "Simulation"])
        self.attack_method_selection.set("Live Network")
        self.attack_method_selection.grid(row=4, column=0, padx=20, pady=(10, 20), sticky="ew")
        
        self.status_label = ctk.CTkLabel(self, text=UIContent.STATUS_READY, font=("Arial", 12))
        self.status_label.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

    def _handle_launch_click(self):
        target_ip = self.ip_entry.get()
        attack_type = self.attack_menu.get()

        if not target_ip:
            self._update_status_label(UIContent.ERROR_NO_IP, "red")
            return
        if attack_type == UIContent.NO_PLAYBOOKS:
            self._update_status_label(UIContent.ERROR_NO_PLAYBOOKS, "red")
            return

        self.launch_button.configure(state="disabled", text=UIContent.LAUNCH_IN_PROGRESS)
        
        threading.Thread(
            target=self._run_attack_and_update_ui,
            args=(target_ip, attack_type),
            daemon=True
        ).start()

    def _run_attack_and_update_ui(self, target_ip, attack_type):
        attack_method = self.attack_method_selection.get()
        self.attack_runner.execute(target_ip, attack_type, attack_method, self.attack_names)
        self.launch_button.configure(state="normal", text=UIContent.LAUNCH_BUTTON)

    def _update_status_label(self, text, color=None):
        if color is None:
            color = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
        self.status_label.configure(text=text, text_color=color)

def main():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = IntruderApp()
    app.mainloop()

if __name__ == "__main__":
    main()