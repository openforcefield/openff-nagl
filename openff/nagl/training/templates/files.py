import importlib_resources

data_directory = importlib_resources.files("openff.nagl") / "training" / "templates"

JINJA_REPORT_TEMPLATE = data_directory / "jinja_report.html"