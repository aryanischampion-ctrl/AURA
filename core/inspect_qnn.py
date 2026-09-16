import onnxruntime as ort
import onnxruntime_qnn as qnn

print("=" * 60)
print("AURA QNN DEVICE DETAILS")
print("=" * 60)

ort.register_execution_provider_library(
    "QNNExecutionProvider",
    qnn.get_library_path()
)

devices = ort.get_ep_devices()

print(f"\nNumber of EP devices: {len(devices)}")

for i, device in enumerate(devices):
    print(f"\n--- Device {i} ---")

    for attr in [
        "ep_name",
        "ep_vendor",
        "ep_device",
        "ep_metadata",
        "vendor",
        "device",
        "metadata",
    ]:
        try:
            value = getattr(device, attr)
            print(f"{attr}: {value}")
        except Exception:
            pass

    try:
        print("repr:", repr(device))
    except Exception:
        pass

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)