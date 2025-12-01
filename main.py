from agent_system.orchestration import run_pipeline

if __name__ == "__main__":
    report = run_pipeline("S001", "T0")
    print(report)
