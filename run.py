from __future__ import annotations

from web_app.app import app


def main() -> None:
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)


if __name__ == "__main__":
    main()
