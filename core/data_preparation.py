import numpy as np
import os
import glob
from scapy.all import PcapReader, IP, TCP, UDP

class DataPreparation:
    """
    A-THENA Data Preparation Module.
    
    This class processes raw PCAP files into model-ready tensors (F, T, Masks, y).
    It supports processing a single file or a directory of files.
    
    IMPORTANT ASSUMPTION:
    All PCAP files provided to an instance of this class are assumed to belong 
    to the SAME class (e.g., all Benign or all Attack). 
    """

    def __init__(self, source_path, d=448, N=30, target_ports=None, 
                 active_flow_threshold=1000, flow_timeout=120):
        """
        Args:
            source_path (str): Path to a single .pcap file OR a directory containing .pcap files.
            d (int): Packet feature length (bytes). Default 448.
            N (int): Maximum flow length (packets). Default 30.
            target_ports (list): Whitelist of ports. None = all.
            active_flow_threshold (int): Active flow count triggering aggregation level shift.
            flow_timeout (float): Seconds of inactivity before a flow is considered closed.
        """
        # Resolve file list
        if os.path.isdir(source_path):
            # Find all .pcap or .pcapng files in the directory
            self.pcap_files = sorted(glob.glob(os.path.join(source_path, "*.pcap*")))
            if not self.pcap_files:
                print(f"Warning: No PCAP files found in {source_path}")
        else:
            self.pcap_files = [source_path]

        self.d = d
        self.N = N
        self.target_ports = set(target_ports) if target_ports else None
        self.threshold = active_flow_threshold
        self.flow_timeout = flow_timeout
        
        # Output storage (accumulates flows from ALL files)
        self.completed_dataset = []

        # Internal state (reset per file)
        self.active_collection = {} 
        self.flow_last_seen = {}
        self.full_flows = set()
        self.aggregation_level = 0 

    def _reset_file_state(self):
        """Resets tracking state for a new PCAP file."""
        self.active_collection = {} 
        self.flow_last_seen = {}
        self.full_flows = set()
        self.aggregation_level = 0 
    
    def _get_flow_key(self, pkt):
        """
        Generates a Bidirectional Flow Key based on aggregation level.
        Ensures A->B and B->A map to the same key by sorting IPs.
        """
        if not pkt.haslayer(IP):
            return None

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto = pkt[IP].proto
        
        sport = 0
        dport = 0
        
        if pkt.haslayer(TCP):
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
            
        # --- Bidirectional Canonicalization ---
        # We sort so that the 'smaller' IP is always first.
        # This groups request/response into the same key.
        if src_ip > dst_ip:
            src_ip, dst_ip = dst_ip, src_ip
            sport, dport = dport, sport

        # --- Adaptive Aggregation Logic ---
        # Level 0: 5-tuple (Specific conversation)
        if self.aggregation_level == 0:
            return (src_ip, dst_ip, sport, dport, proto)
        
        # Level 1: 3-tuple (Ignore Ports - Host-to-Host traffic)
        elif self.aggregation_level == 1:
            return (src_ip, dst_ip, proto)
        
        # Level 2: 2-tuple (Ignore Source IP - Traffic Destined to Host)
        # TODO: Decide explicitly which IP address to retain here. In attack
        #       scenarios you must ensure that the victim's IP is the one kept.
        #       This logic may require overriding the default (dst_ip) selection.
        elif self.aggregation_level == 2:
            return (dst_ip, proto)
        
        return None

    def _maintenance_and_adaptive_check(self, current_time):
        """
        Performs two tasks:
        1. Cleans up timed-out flows from tracking structures.
        2. Adjusts aggregation level based on 'true' active flow volume.
        """
        # --- 1. Cleanup Timeouts ---
        # Identify keys to remove
        keys_to_remove = []
        for key, last_seen in self.flow_last_seen.items():
            if (current_time - last_seen) > self.flow_timeout:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.flow_last_seen[key]
            
            # Also remove from state trackers if present
            if key in self.full_flows:
                self.full_flows.remove(key)
            if key in self.active_collection:
                # If a flow times out while collecting, we treat it as a finished sample
                self.completed_dataset.append(self.active_collection.pop(key))

        # --- 2. Adaptive Aggregation Update ---
        # The volume is the count of valid keys in flow_last_seen
        current_volume = len(self.flow_last_seen)
        
        if current_volume > (self.threshold * 5):
            self.aggregation_level = 2
        elif current_volume > self.threshold:
            self.aggregation_level = 1
        else:
            self.aggregation_level = 0

    def packet_filtering(self, pkt):
        """
        TODO: CUSTOMIZE THIS LOGIC BASED ON YOUR NETWORK CONTEXT.
        
        Filtering should operate on two distinct levels:
        
        1. Protocol Selection:
           - Filter based on protocol susceptibility (e.g., HTTP, ARP, ICMP).
           - Example: Retaining HTTP allows inspection of XSS, SQLi, and DoS.
           - Current Code: Implements port-based whitelisting (self.target_ports).
             You may need to add checks for non-port protocols like `pkt.haslayer(ARP)`.
           
        2. Network-Specific Constraints:
           - Target precise communication channels based on topology.
           - Example: For a brute-force attack, configure the filter to isolate traffic 
             exclusively between the attacker's suspected subnet and the victim's IP.
           - Goal: Eliminate background noise and ensure the model processes only 
             traffic pertinent to the anticipated attack vector.
        """
        # 1. Basic Protocol Check (Must be IP for this implementation, 
        #    but you can enable ARP/ICMP here if your model handles them)
        if not pkt.haslayer(IP):
            return False
            
        # 2. Port/Service Filtering (Protocol Selection)
        if self.target_ports:
            sport = pkt[TCP].sport if pkt.haslayer(TCP) else (pkt[UDP].sport if pkt.haslayer(UDP) else 0)
            dport = pkt[TCP].dport if pkt.haslayer(TCP) else (pkt[UDP].dport if pkt.haslayer(UDP) else 0)
            
            # Keep if either source or dest matches target services
            if (sport in self.target_ports) or (dport in self.target_ports):
                return True
            return False
            
        # 3. Network-Specific Constraints (Example Placeholder)
        # if pkt[IP].src == "192.168.1.100" and pkt[IP].dst == "10.0.0.5":
        #     return True
        
        return True
    
    def packet_preprocessing(self, pkt):
        """
        Header removal (Eth + IPs), truncation/padding, normalization.
        """
        # 1. Get raw bytes of the IP layer (strips Ethernet automatically)
        # We work with the IP layer down to payload
        try:
            raw_bytes = bytearray(bytes(pkt[IP]))
        except Exception:
            return np.zeros(self.d, dtype=np.float32)
        
        # 2. Remove Source and Dest IP addresses
        # IPv4 Header standard: Src is bytes 12-15, Dst is 16-19.
        # Note: We delete higher index first to not shift the lower index.
        try:
            if len(raw_bytes) >= 20:
                del raw_bytes[16:20] 
                del raw_bytes[12:16] 
        except IndexError:
            pass 
            
        # 3. Truncate or Pad to length d
        # Convert to list of integers for normalization
        byte_list = list(raw_bytes)
        if len(byte_list) > self.d:
            byte_list = byte_list[:self.d]
        else:
            byte_list += [0] * (self.d - len(byte_list))
            
        # 4. Normalize to [0, 1]
        normalized_vector = np.array(byte_list, dtype=np.float32) / 255.0
        
        return normalized_vector

    def _process_single_file(self, file_path, limit=None):
        """
        Logic to process one PCAP file.
        Args:
            limit (int): Maximum total flows to collect (global across files).
                         If None, no limit.
        """
        packet_count = 0
        
        with PcapReader(file_path) as pcap_reader:
            for pkt in pcap_reader:

                # Stop reading if we hit the limit
                if limit is not None and len(self.completed_dataset) >= limit:
                    return 
                
                packet_count += 1
                
                # 1. Filter
                if not self.packet_filtering(pkt):
                    continue

                arrival_time = float(pkt.time)
                
                # 2. Maintenance
                if packet_count % 100 == 0:
                    self._maintenance_and_adaptive_check(arrival_time)
                
                # 3. Flow Identification
                flow_key = self._get_flow_key(pkt)
                if flow_key is None:
                    continue
                
                self.flow_last_seen[flow_key] = arrival_time
                
                if flow_key in self.full_flows:
                    continue

                # 4. Data Collection
                if flow_key not in self.active_collection:
                    self.active_collection[flow_key] = {
                        'packets': [],
                        'timestamps': [],
                        'start_time': arrival_time
                    }
                
                relative_time = arrival_time - self.active_collection[flow_key]['start_time']
                processed_pkt = self.packet_preprocessing(pkt)
                
                self.active_collection[flow_key]['packets'].append(processed_pkt)
                self.active_collection[flow_key]['timestamps'].append(relative_time)
                
                if len(self.active_collection[flow_key]['packets']) == self.N:
                    self.completed_dataset.append(self.active_collection.pop(flow_key))
                    self.full_flows.add(flow_key)

        # End of File: Flush remaining incomplete flows
        # Check limit again before flushing to avoid overshooting too much
        remaining_slots = float('inf') if limit is None else (limit - len(self.completed_dataset))
        
        if remaining_slots > 0:
            count_flushed = 0
            for flow_data in self.active_collection.values():
                if count_flushed >= remaining_slots:
                    break
                self.completed_dataset.append(flow_data)
                count_flushed += 1

    def run_pipeline(self, label=None, limit=None):
        """
        Executes the pipeline on all configured files.
        
        Args:
            label (int, optional): The integer class label.
            limit (int, optional): Max number of flows to extract.
        
        Returns:
            F, T, Masks, [y]
        """
        self.completed_dataset = [] # Reset output storage
        
        for pcap_file in self.pcap_files:

            # Stop completely if we already hit the limit from previous files in this directory
            if limit is not None and len(self.completed_dataset) >= limit:
                break

            # Reset ephemeral state to prevent timestamp conflicts between files
            self._reset_file_state()
            self._process_single_file(pcap_file, limit)
            
        return self._finalize_dataset(label)

    def _finalize_dataset(self, label):
        """
        Convert to Numpy with Zero Padding, Masks, and optional Labels.
        """
        num_flows = len(self.completed_dataset)

        if num_flows > 0:
            # Calculate true lengths before padding
            lengths = [len(f['packets']) for f in self.completed_dataset]
            l_min = np.min(lengths)
            l_max = np.max(lengths)
            l_mean = np.mean(lengths)
            
            print(f" -> Extracted {num_flows} flows.")
            print(f"    Stats: Min Length: {l_min}, Max Length: {l_max}, Average Length: {l_mean:.2f}")
        else:
            print(" -> No valid flows found.")
        
        F = np.zeros((num_flows, self.N, self.d), dtype=np.float32)
        T = np.zeros((num_flows, self.N), dtype=np.float32)
        Masks = np.zeros((num_flows, self.N), dtype=np.float32)
        
        for i, flow in enumerate(self.completed_dataset):
            pkts = flow['packets']
            times = flow['timestamps']
            length = len(pkts) # Actual length (<= N)
            
            # Fill Data (Vectors)
            F[i, :length, :] = np.array(pkts)
            
            # Fill Timestamps
            T[i, :length] = np.array(times)
            
            # Fill Masks (1 = Real Data, 0 = Padding)
            Masks[i, :length] = 1.0
        
        if label is not None:
            # Generate label array for this homogeneous batch
            y = np.full((num_flows,), label, dtype=np.int32)
            return F, T, Masks, y
        
        return F, T, Masks