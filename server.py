from waitress import serve
import webbrowser

from app import init_dashboard

app = init_dashboard()

webbrowser.open("http://localhost:8049")
serve(app.server, port=8049)