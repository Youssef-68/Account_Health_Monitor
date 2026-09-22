from src.data_loader import load_data
from src.health_engine import get_health_summary

def main():
    print("=== Account Health Monitor - CLI Batch Assessor ===")
    df = load_data()
    summary = get_health_summary(df)

    print("\n--- Summary Report ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()