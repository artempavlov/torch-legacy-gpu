"""Small GitHub API client; the build process never receives the write token."""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import http.client


class GitHub:
    def __init__(self, repository=None):
        self.repository = repository or os.environ.get("GITHUB_REPOSITORY")
        self.token = os.environ.get("GH_TOKEN", "")

    def request(self, path, method="GET", data=None, content_type="application/json"):
        url = path if path.startswith("https://") else "https://api.github.com" + path
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "torch-legacy-gpu"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if data is not None:
            headers["Content-Type"] = content_type
            if not isinstance(data, bytes):
                data = json.dumps(data).encode()
        for attempt in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, data, headers, method=method), timeout=120) as response:
                    body = response.read()
                    return json.loads(body) if body else None
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))

    def release(self, tag):
        try:
            return self.request(f"/repos/{self.repository}/releases/tags/{tag}")
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            # Draft releases are not always returned by the tags endpoint.
            for page in range(1, 11):
                releases = self.request(f"/repos/{self.repository}/releases?per_page=100&page={page}")
                for release in releases:
                    if release["tag_name"] == tag:
                        return release
                if len(releases) < 100:
                    return None
            raise RuntimeError("Release pagination limit reached")

    def assets(self, release):
        result = []
        for page in range(1, 11):
            batch = self.request(f"/repos/{self.repository}/releases/{release['id']}/assets?per_page=100&page={page}")
            result.extend(batch)
            if len(batch) < 100:
                return result
        raise RuntimeError("Asset pagination limit reached")

    def upload(self, release, path):
        if path.stat().st_size >= 2 * 1024 ** 3:
            raise RuntimeError(f"GitHub's 2 GiB asset limit exceeded: {path.name}")
        name = urllib.parse.quote(path.name)
        url = release["upload_url"].split("{")[0] + "?name=" + name
        parsed = urllib.parse.urlsplit(url)
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=600)
        try:
            connection.putrequest("POST", parsed.path + "?" + parsed.query)
            connection.putheader("Authorization", "Bearer " + self.token)
            connection.putheader("User-Agent", "torch-legacy-gpu")
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("Content-Length", str(path.stat().st_size))
            connection.endheaders()
            with path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    connection.send(block)
            response = connection.getresponse()
            body = response.read()
            if response.status != 201:
                raise RuntimeError(f"Upload {path.name}: HTTP {response.status}: {body[:500]!r}")
            return json.loads(body)
        finally:
            connection.close()
