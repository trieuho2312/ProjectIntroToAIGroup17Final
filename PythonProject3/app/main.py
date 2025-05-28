import streamlit.web.cli as stcli
import sys

def run():
    sys.argv = ["streamlit", "run", "app/app.py"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    run()
