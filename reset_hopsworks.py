import hopsworks
from datetime import datetime
import sys


def reset_hopsworks():
    """Delete all feature groups and models to start fresh"""
    print("\n=== RESETTING HOPSWORKS PROJECT ===")
    print(f"Current time (UTC): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"User: humaila0")

    try:
        print("\nConnecting to Hopsworks...")
        project = hopsworks.login()
        print(f"Connected to project: {project.name}")
    except Exception as e:
        print(f"Error connecting to Hopsworks: {str(e)}")
        sys.exit(1)

    # Delete feature groups
    try:
        print("\nDeleting feature groups...")
        fs = project.get_feature_store()

        # Retrieve all feature groups
        feature_groups = fs.get_feature_groups()
        for fg in feature_groups:
            try:
                print(f"Deleting feature group: {fg.name} (version {fg.version})...")
                fg.delete()
                print(f"✅ Deleted feature group: {fg.name} (version {fg.version})")
            except Exception as e:
                print(f"Error deleting feature group {fg.name} (version {fg.version}): {str(e)}")
    except Exception as e:
        print(f"Error retrieving feature groups: {str(e)}")

    # Delete models
    try:
        print("\nDeleting models...")
        mr = project.get_model_registry()

        # Retrieve all models dynamically
        models = mr.get_models_summary()
        for model_summary in models:
            try:
                model = mr.get_model(model_summary['name'], version=model_summary['version'])
                print(f"Deleting model: {model.name} (version {model.version})...")
                model.delete()
                print(f"✅ Deleted model: {model.name} (version {model.version})")
            except Exception as e:
                print(f"Error deleting model {model_summary['name']} (version {model_summary['version']}): {str(e)}")
    except Exception as e:
        print(f"Error retrieving models: {str(e)}")

    print("\n=== HOPSWORKS RESET COMPLETE ===")


if __name__ == "__main__":
    # Ask for confirmation
    print("WARNING: This will delete all feature groups and models in your Hopsworks project.")
    confirmation = input("Type 'YES' to confirm: ")

    if confirmation == "YES":
        reset_hopsworks()
    else:
        print("Reset canceled.")
        sys.exit(0)