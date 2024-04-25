import importlib.resources

data_directory = importlib.resources.files("openff.nagl") / "training" / "templates"

JINJA_REPORT_TEMPLATE = data_directory / "jinja_report.html"