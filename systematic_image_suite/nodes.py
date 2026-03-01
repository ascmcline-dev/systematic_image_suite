"""
Systematic Image Suite — ComfyUI custom nodes

Fix v0.1.1 (key change):
- Do NOT rely on UNIQUE_ID for pairing (can differ at runtime).
- Derive node_ids from hidden PROMPT graph keys (these are what connections actually reference).
- Stepper finds its own node_id by matching inputs, then finds the Loader consuming it.
- Loader also resolves its own node_id and uses it for persistent state key.

Nodes:
A) SystematicImageLoader
B) ImageListStepper
C) AutoQueueNextRun
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except Exception:
    folder_paths = None  # type: ignore


# ----------------------------
# Helpers
# ----------------------------

def _norm_path(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return ""
    p = os.path.expandvars(os.path.expanduser(p))
    return os.path.abspath(p)


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _get_temp_dir() -> str:
    if folder_paths is not None:
        try:
            return folder_paths.get_temp_directory()
        except Exception:
            pass
    return os.path.abspath(os.path.join(os.getcwd(), "temp"))


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class _FileLock:
    def __init__(self, lock_path: str, timeout_s: float = 5.0, poll_s: float = 0.05):
        self.lock_path = lock_path
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._acquired = False

    def acquire(self) -> None:
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, str(os.getpid()).encode("utf-8"))
                finally:
                    os.close(fd)
                self._acquired = True
                return
            except FileExistsError:
                time.sleep(self.poll_s)
        raise TimeoutError(f"Timed out acquiring lock: {self.lock_path}")

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass
        self._acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def _parse_extensions(extensions_csv: str) -> List[str]:
    exts: List[str] = []
    for part in (extensions_csv or "").split(","):
        e = part.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        exts.append(e)
    return exts or [".png", ".jpg", ".jpeg", ".webp"]


def _discover_files(folder_path: str, recursive: bool, extensions_csv: str, sort_mode: str, filter_contains: str) -> List[str]:
    folder_path = _norm_path(folder_path)
    if not folder_path or not os.path.isdir(folder_path):
        return []

    exts = set(_parse_extensions(extensions_csv))
    filt = (filter_contains or "").strip().lower()

    files: List[str] = []
    if recursive:
        for root, _dirs, fnames in os.walk(folder_path):
            for fn in fnames:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in exts:
                    continue
                full = os.path.abspath(os.path.join(root, fn))
                if filt and (filt not in full.lower()):
                    continue
                files.append(full)
    else:
        for fn in os.listdir(folder_path):
            full = os.path.abspath(os.path.join(folder_path, fn))
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in exts:
                continue
            if filt and (filt not in full.lower()):
                continue
            files.append(full)

    if sort_mode == "mtime":
        files.sort(key=lambda p: os.path.getmtime(p))
    elif sort_mode == "fullpath":
        files.sort(key=lambda p: p.lower())
    else:
        files.sort(key=lambda p: os.path.basename(p).lower())

    return files


def _list_hash(paths: List[str]) -> str:
    return _sha256_text("\n".join(paths))


def _load_image_as_comfy_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _apply_wrap_clamp(i: int, count: int, wrap_mode: str) -> int:
    if count <= 0:
        return 0
    if wrap_mode == "wrap":
        return int(i) % int(count)
    if i < 0:
        return 0
    if i >= count:
        return count - 1
    return int(i)


def _extract_prompt_graph(prompt: Any) -> Dict[str, Any]:
    """
    PROMPT may be either:
    - dict of node_id -> node_data
    - or dict with key "prompt" containing that dict
    """
    if isinstance(prompt, dict) and "prompt" in prompt and isinstance(prompt["prompt"], dict):
        return prompt["prompt"]
    if isinstance(prompt, dict):
        return prompt
    return {}


def _get_literal_input(inputs: Dict[str, Any], key: str, default: Any) -> Any:
    """
    If an input is connected, it'll be [node_id, out_index]. We only use literal widget values.
    """
    v = inputs.get(key, default)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return default
    return v


def _matches_stepper_inputs(node_inputs: Dict[str, Any], advance: bool, delta: int, reset_to_start: bool, jump_to_index: int) -> bool:
    try:
        adv = bool(_get_literal_input(node_inputs, "advance", True))
        dlt = int(_get_literal_input(node_inputs, "delta", 1))
        rst = bool(_get_literal_input(node_inputs, "reset_to_start", False))
        jmp = int(_get_literal_input(node_inputs, "jump_to_index", -1))
        return (adv == bool(advance) and dlt == int(delta) and rst == bool(reset_to_start) and jmp == int(jump_to_index))
    except Exception:
        return False


def _matches_loader_inputs(node_inputs: Dict[str, Any], folder_path: str, recursive: bool, extensions_csv: str, sort_mode: str, wrap_mode: str, filter_contains: str) -> bool:
    try:
        fp = str(_get_literal_input(node_inputs, "folder_path", r"C:\path\to\images"))
        rec = bool(_get_literal_input(node_inputs, "recursive", False))
        ex = str(_get_literal_input(node_inputs, "extensions_csv", ".png,.jpg,.jpeg,.webp"))
        sm = str(_get_literal_input(node_inputs, "sort_mode", "name"))
        wm = str(_get_literal_input(node_inputs, "wrap_mode", "clamp"))
        fc = str(_get_literal_input(node_inputs, "filter_contains", ""))

        # normalize key fields
        return (
            _norm_path(fp).lower() == _norm_path(folder_path).lower()
            and rec == bool(recursive)
            and ex.strip().lower() == (extensions_csv or "").strip().lower()
            and sm == str(sort_mode)
            and wm == str(wrap_mode)
            and fc.strip().lower() == (filter_contains or "").strip().lower()
        )
    except Exception:
        return False


def _find_stepper_node_id(prompt_graph: Dict[str, Any], advance: bool, delta: int, reset_to_start: bool, jump_to_index: int) -> Optional[str]:
    """
    Find the actual node_id key in PROMPT for THIS stepper, by matching its inputs.
    """
    for node_id, node_data in prompt_graph.items():
        if not isinstance(node_data, dict):
            continue
        if node_data.get("class_type") != "ImageListStepper":
            continue
        inputs = node_data.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if _matches_stepper_inputs(inputs, advance, delta, reset_to_start, jump_to_index):
            return str(node_id)
    return None


def _find_connected_loader_id(prompt_graph: Dict[str, Any], stepper_node_id: str) -> Optional[str]:
    """
    Find the loader node that consumes [stepper_node_id, out_index] in any of its inputs.
    """
    for node_id, node_data in prompt_graph.items():
        if not isinstance(node_data, dict):
            continue
        if node_data.get("class_type") != "SystematicImageLoader":
            continue
        inputs = node_data.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for v in inputs.values():
            if isinstance(v, (list, tuple)) and len(v) == 2 and str(v[0]) == str(stepper_node_id):
                return str(node_id)
    return None


def _find_loader_node_id(prompt_graph: Dict[str, Any], folder_path: str, recursive: bool, extensions_csv: str, sort_mode: str, wrap_mode: str, filter_contains: str) -> Optional[str]:
    """
    Find this loader node_id by matching its key inputs. (Used by Loader itself.)
    """
    for node_id, node_data in prompt_graph.items():
        if not isinstance(node_data, dict):
            continue
        if node_data.get("class_type") != "SystematicImageLoader":
            continue
        inputs = node_data.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if _matches_loader_inputs(inputs, folder_path, recursive, extensions_csv, sort_mode, wrap_mode, filter_contains):
            return str(node_id)
    return None


@dataclass
class LoaderState:
    index: int = 0
    step_signal: int = 0
    list_hash: str = ""


class _StateDB:
    def __init__(self, namespace: str = "systematic_image_suite"):
        temp_dir = _get_temp_dir()
        _safe_mkdir(temp_dir)
        self.path = os.path.join(temp_dir, f"{namespace}_state.json")
        self.lock = self.path + ".lock"

    def _read(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"version": 1, "loaders": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                return {"version": 1, "loaders": {}}
            if "loaders" not in d or not isinstance(d["loaders"], dict):
                d["loaders"] = {}
            return d
        except Exception:
            return {"version": 1, "loaders": {}}

    def _write_atomic(self, d: Dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def get(self, key: str) -> LoaderState:
        with _FileLock(self.lock):
            d = self._read()
            e = d.get("loaders", {}).get(key)
            if isinstance(e, dict):
                return LoaderState(
                    index=int(e.get("index", 0)),
                    step_signal=int(e.get("step_signal", 0)),
                    list_hash=str(e.get("list_hash", "")),
                )
            return LoaderState()

    def set(self, key: str, state: LoaderState) -> None:
        with _FileLock(self.lock):
            d = self._read()
            loaders = d.get("loaders", {})
            loaders[key] = {
                "index": int(state.index),
                "step_signal": int(state.step_signal),
                "list_hash": str(state.list_hash),
                "updated": time.time(),
            }
            d["loaders"] = loaders
            self._write_atomic(d)


_STATE = _StateDB()


def _loader_key(
    loader_node_id: str,
    folder_path: str,
    recursive: bool,
    extensions_csv: str,
    sort_mode: str,
    wrap_mode: str,
    filter_contains: str,
) -> str:
    payload = {
        "loader_node_id": str(loader_node_id),
        "folder_path": _norm_path(folder_path).lower(),
        "recursive": bool(recursive),
        "extensions_csv": (extensions_csv or "").strip().lower(),
        "sort_mode": str(sort_mode),
        "wrap_mode": str(wrap_mode),
        "filter_contains": (filter_contains or "").strip().lower(),
    }
    return _sha256_text(_json_dumps_stable(payload))


def _post_json(url: str, payload: Dict[str, Any], timeout_s: float = 2.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except Exception:
        return {"error": "Non-JSON response", "raw": raw}


# ----------------------------
# Node A — Systematic Image Loader
# ----------------------------

class SystematicImageLoader:
    CATEGORY = "utils/systematic"
    FUNCTION = "run"

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "INT", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("image", "save_name", "index", "count", "remaining", "done", "full_path", "list_hash")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": r"C:\path\to\images"}),
                "filename_prefix": ("STRING", {"default": "batch"}),
                "step_signal": ("INT", {"default": 0, "min": 0, "max": 2_000_000, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 2_000_000, "step": 1}),
                "reset_list": ("BOOLEAN", {"default": False}),
                "recursive": ("BOOLEAN", {"default": False}),
                "extensions_csv": ("STRING", {"default": ".png,.jpg,.jpeg,.webp"}),
                "sort_mode": (["name", "fullpath", "mtime"], {"default": "name"}),
                "wrap_mode": (["clamp", "wrap"], {"default": "clamp"}),
                "filter_contains": ("STRING", {"default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(
        self,
        folder_path: str,
        filename_prefix: str,
        step_signal: int,
        start_index: int,
        reset_list: bool,
        recursive: bool,
        extensions_csv: str,
        sort_mode: str,
        wrap_mode: str,
        filter_contains: str,
        prompt: Any = None,
    ):
        prompt_graph = _extract_prompt_graph(prompt)
        loader_node_id = _find_loader_node_id(prompt_graph, folder_path, recursive, extensions_csv, sort_mode, wrap_mode, filter_contains) or "loader_unknown"

        paths = _discover_files(folder_path, recursive, extensions_csv, sort_mode, filter_contains)
        count = len(paths)
        lhash = _list_hash(paths)

        key = _loader_key(loader_node_id, folder_path, recursive, extensions_csv, sort_mode, wrap_mode, filter_contains)
        st = _STATE.get(key)

        if reset_list or (st.list_hash != lhash):
            st.index = int(start_index)
            st.list_hash = lhash

        idx = _apply_wrap_clamp(st.index, count, wrap_mode)
        full_path = paths[idx] if count > 0 else ""

        done = bool(count == 0 or (wrap_mode == "clamp" and idx >= count - 1))
        remaining = int(max(0, count - idx - 1))

        save_name = f"{filename_prefix}_{idx:06d}"

        if full_path and os.path.exists(full_path):
            image = _load_image_as_comfy_tensor(full_path)
        else:
            image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        st.index = int(idx)
        st.list_hash = lhash
        _STATE.set(key, st)

        return (image, save_name, int(idx), int(count), int(remaining), bool(done), str(full_path), str(lhash))


# ----------------------------
# Node B — Stepper
# ----------------------------

class ImageListStepper:
    CATEGORY = "utils/systematic"
    FUNCTION = "run"

    RETURN_TYPES = ("INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("step_signal", "index", "count", "done")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "advance": ("BOOLEAN", {"default": True}),
                "delta": ("INT", {"default": 1, "min": -9999, "max": 9999, "step": 1}),
                "reset_to_start": ("BOOLEAN", {"default": False}),
                "jump_to_index": ("INT", {"default": -1, "min": -1, "max": 2_000_000, "step": 1}),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(
        self,
        advance: bool,
        delta: int,
        reset_to_start: bool,
        jump_to_index: int,
        prompt: Any = None,
    ):
        prompt_graph = _extract_prompt_graph(prompt)

        stepper_node_id = _find_stepper_node_id(prompt_graph, advance, delta, reset_to_start, jump_to_index)
        if stepper_node_id is None:
            # Could not identify self in graph (rare). Still emit changing step_signal but no pairing.
            # Return index=-1 to make failure obvious.
            return (int(time.time() * 1000) % 2_000_000, -1, 0, True)

        loader_id = _find_connected_loader_id(prompt_graph, stepper_node_id)
        if loader_id is None:
            return (int(time.time() * 1000) % 2_000_000, -1, 0, True)

        loader_node = prompt_graph.get(loader_id, {})
        inputs = loader_node.get("inputs", {}) if isinstance(loader_node, dict) else {}
        if not isinstance(inputs, dict):
            inputs = {}

        folder_path = str(_get_literal_input(inputs, "folder_path", r"C:\path\to\images"))
        recursive = bool(_get_literal_input(inputs, "recursive", False))
        extensions_csv = str(_get_literal_input(inputs, "extensions_csv", ".png,.jpg,.jpeg,.webp"))
        sort_mode = str(_get_literal_input(inputs, "sort_mode", "name"))
        wrap_mode = str(_get_literal_input(inputs, "wrap_mode", "clamp"))
        filter_contains = str(_get_literal_input(inputs, "filter_contains", ""))
        start_index = int(_get_literal_input(inputs, "start_index", 0))

        paths = _discover_files(folder_path, recursive, extensions_csv, sort_mode, filter_contains)
        count = len(paths)
        lhash = _list_hash(paths)

        key = _loader_key(loader_id, folder_path, recursive, extensions_csv, sort_mode, wrap_mode, filter_contains)
        st = _STATE.get(key)

        # If list changed, reset index to start_index; do NOT advance on this same run.
        fresh_reset = False
        if st.list_hash != lhash:
            st.index = int(start_index)
            st.list_hash = lhash
            fresh_reset = True

        new_index = int(st.index)
        if reset_to_start:
            new_index = int(start_index)
        elif int(jump_to_index) >= 0:
            new_index = int(jump_to_index)
        elif advance and not fresh_reset:
            new_index = int(new_index) + int(delta)

        new_index = _apply_wrap_clamp(new_index, count, wrap_mode)

        st.index = int(new_index)
        st.list_hash = lhash
        st.step_signal = int(st.step_signal) + 1
        _STATE.set(key, st)

        done = bool(count == 0 or (wrap_mode == "clamp" and new_index >= count - 1))
        return (int(st.step_signal), int(new_index), int(count), bool(done))


# ----------------------------
# Node C — Auto Queue Next Run (Loop)
# ----------------------------

class _AutoQueueDB:
    def __init__(self, namespace: str = "systematic_image_suite_autoqueue"):
        temp_dir = _get_temp_dir()
        _safe_mkdir(temp_dir)
        self.path = os.path.join(temp_dir, f"{namespace}_state.json")
        self.lock = self.path + ".lock"

    def _read(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"version": 1, "items": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                return {"version": 1, "items": {}}
            if "items" not in d or not isinstance(d["items"], dict):
                d["items"] = {}
            return d
        except Exception:
            return {"version": 1, "items": {}}

    def _write_atomic(self, d: Dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def get(self, key: str) -> Dict[str, Any]:
        with _FileLock(self.lock):
            d = self._read()
            return dict(d.get("items", {}).get(key, {}))

    def set(self, key: str, item: Dict[str, Any]) -> None:
        with _FileLock(self.lock):
            d = self._read()
            items = d.get("items", {})
            items[key] = dict(item)
            items[key]["updated"] = time.time()
            d["items"] = items
            self._write_atomic(d)


_AUTOQ = _AutoQueueDB()


class AutoQueueNextRun:
    CATEGORY = "utils/systematic"
    FUNCTION = "run"

    RETURN_TYPES = ("BOOLEAN", "STRING", "INT", "STRING")
    RETURN_NAMES = ("queued", "prompt_id", "queue_number", "error")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False}),
                "server_address": ("STRING", {"default": "127.0.0.1:8188"}),
                "client_id": ("STRING", {"default": "systematic_image_suite"}),
                "step_signal": ("INT", {"default": 0, "min": 0, "max": 2_000_000, "step": 1}),
                "done": ("BOOLEAN", {"default": False}),
                "stop_when_done": ("BOOLEAN", {"default": True}),
                "max_loops": ("INT", {"default": 0, "min": 0, "max": 10_000_000, "step": 1}),
                "reset_loop_state": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "trigger_image": ("IMAGE",),
                "trigger_latent": ("LATENT",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(
        self,
        enabled: bool,
        server_address: str,
        client_id: str,
        step_signal: int,
        done: bool,
        stop_when_done: bool,
        max_loops: int,
        reset_loop_state: bool,
        trigger_image=None,
        trigger_latent=None,
        prompt: Any = None,
        unique_id: Any = None,
    ):
        if not enabled:
            return (False, "", -1, "")

        if stop_when_done and bool(done):
            return (False, "", -1, "")

        prompt_graph = _extract_prompt_graph(prompt)
        if not isinstance(prompt_graph, dict) or len(prompt_graph) == 0:
            return (False, "", -1, "PROMPT graph missing/invalid; cannot auto-queue")

        key = _sha256_text(_json_dumps_stable({
            "autoqueue_node": str(unique_id),
            "server": str(server_address).strip(),
            "client_id": str(client_id).strip(),
        }))

        st = _AUTOQ.get(key)
        if reset_loop_state:
            st = {}

        last_sig = int(st.get("last_step_signal", -1))
        loops = int(st.get("loops", 0))

        if int(step_signal) <= last_sig:
            return (False, "", -1, "")

        if int(max_loops) > 0 and loops >= int(max_loops):
            return (False, "", -1, f"max_loops reached ({max_loops})")

        addr = str(server_address).strip()
        if addr.startswith("http://") or addr.startswith("https://"):
            base = addr
        else:
            base = "http://" + addr
        url = base.rstrip("/") + "/prompt"

        payload = {"prompt": prompt_graph, "client_id": str(client_id).strip() or "systematic_image_suite"}

        try:
            resp = _post_json(url, payload, timeout_s=2.0)
        except Exception as e:
            return (False, "", -1, f"POST /prompt failed: {e}")

        prompt_id = str(resp.get("prompt_id", ""))
        number = resp.get("number", -1)

        st["last_step_signal"] = int(step_signal)
        st["loops"] = loops + 1
        _AUTOQ.set(key, st)

        if "error" in resp and resp["error"]:
            return (False, prompt_id, int(number) if isinstance(number, int) else -1, str(resp["error"]))

        return (True, prompt_id, int(number) if isinstance(number, int) else -1, "")
