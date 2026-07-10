import os
import json
import pytest
import tempfile
import socket
import urllib.error
import urllib.parse
from unittest.mock import patch, MagicMock
import xml.etree.ElementTree as ET

from scripts.download_papers import (
    extract_arxiv_id,
    load_registry,
    save_registry,
    make_request_with_retry,
    download_arxiv_papers
)

# Set up test suite for stress-testing and boundary verification

def test_stress_query_encoding():
    """Verify that query parameters, including empty and special character queries, are correctly encoded."""
    # Test special characters
    special_query = 'all:"AI Agents" & category:cs.AI'
    encoded = urllib.parse.quote(special_query)
    assert encoded == "all%3A%22AI%20Agents%22%20%26%20category%3Acs.AI"
    
    # Test empty query
    empty_query = ""
    encoded_empty = urllib.parse.quote(empty_query)
    assert encoded_empty == ""

@patch("scripts.download_papers.make_request_with_retry")
def test_stress_empty_query_execution(mock_make_request):
    """Verify that executing download_papers with an empty query (which returns empty entries) handles gracefully and returns."""
    empty_xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>ArXiv Query: </title>
    </feed>
    """
    mock_make_request.return_value = empty_xml.encode('utf-8')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "sources")
        registry_path = os.path.join(output_dir, "registry.json")
        
        res = download_arxiv_papers(
            query="",
            max_results=5,
            output_dir=output_dir,
            registry_path=registry_path
        )
        assert isinstance(res, dict)

def test_stress_invalid_directory():
    """Verify how download_papers handles invalid directories (like using null byte or illegal chars)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Null bytes or illegal paths on Windows/Linux usually raise ValueError or OSError
        invalid_dir = os.path.join(tmpdir, "invalid_\0_dir")
        registry_path = os.path.join(tmpdir, "registry.json")
        
        with pytest.raises((OSError, ValueError)):
            download_arxiv_papers(
                query="all:AI",
                max_results=1,
                output_dir=invalid_dir,
                registry_path=registry_path
            )

@patch("time.sleep", return_value=None)
@patch("urllib.request.urlopen")
def test_stress_socket_timeout_recovery(mock_urlopen, mock_sleep):
    """Verify that socket timeouts are caught and retried, and recover if a subsequent attempt succeeds."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = b"<feed>success</feed>"
    
    # Fail first with TimeoutError, then succeed
    mock_urlopen.side_effect = [
        TimeoutError("Connection timed out"),
        mock_resp
    ]
    
    data = make_request_with_retry("http://test.url", is_download=False)
    assert data == b"<feed>success</feed>"
    assert mock_urlopen.call_count == 2
    mock_sleep.assert_any_call(3) # Wait 3s after first failure

@patch("time.sleep", return_value=None)
@patch("urllib.request.urlopen")
def test_stress_socket_timeout_exhaustion(mock_urlopen, mock_sleep):
    """Verify that if socket timeouts persist, make_request_with_retry eventually reraises the exception."""
    mock_urlopen.side_effect = TimeoutError("Connection timed out")
    
    with pytest.raises(TimeoutError):
        make_request_with_retry("http://test.url", is_download=False)
        
    assert mock_urlopen.call_count == 3
    mock_sleep.assert_any_call(3)
    mock_sleep.assert_any_call(6)

@patch("time.sleep", return_value=None)
@patch("urllib.request.urlopen")
def test_stress_rate_limit_delay_enforcement(mock_urlopen, mock_sleep):
    """Verify that 3-second delay is enforced after every successful request (metadata query)."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = b"<feed>success</feed>"
    mock_urlopen.return_value = mock_resp
    
    # Perform a query
    make_request_with_retry("http://test.url", is_download=False)
    
    # Must sleep 3 seconds on success
    mock_sleep.assert_called_once_with(3)

@patch("urllib.request.urlopen")
@patch("urllib.request.urlretrieve")
def test_stress_download_failures_no_rate_limit_sleep(mock_urlretrieve, mock_urlopen):
    """Verify that when paper downloads fail after all attempts, no extra 3-second delay is added between failures in the main loop."""
    mock_xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <title>Paper 1</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00001" />
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2401.00002v1</id>
        <title>Paper 2</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00002" />
      </entry>
    </feed>
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "sources")
        registry_path = os.path.join(output_dir, "registry.json")
        
        # Mock XML query succeeds
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.read.return_value = mock_xml.encode('utf-8')
        mock_urlopen.return_value = mock_resp
        
        # Both downloads fail completely
        mock_urlretrieve.side_effect = urllib.error.HTTPError("url", 404, "Not Found", None, None)
        
        # We track sleeps during the download paper flow
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError) as excinfo:
                download_arxiv_papers(
                    query="all:AI",
                    max_results=2,
                    output_dir=output_dir,
                    registry_path=registry_path
                )
            
            assert "downloads failed" in str(excinfo.value)
            
            # Since mock_urlretrieve raised HTTPError, make_request_with_retry reraises on last attempt.
            # In download_arxiv_papers:
            # - query success: sleeps 3s inside make_request_with_retry.
            # - download 1: fail inside make_request_with_retry (retries 3 times -> sleeps 3s, 6s). No success sleep.
            # - download 2: fail inside make_request_with_retry (retries 3 times -> sleeps 3s, 6s). No success sleep.
            # Total sleeps: 3s (success), and four backoffs (3s, 6s, 3s, 6s).
            # No extra delay is introduced in the download loop itself on failure.
            assert mock_sleep.call_count == 5

def test_stress_invalid_xml_feed():
    """Verify that download_papers handles malformed XML gracefully by logging a critical error and exiting with 1 (verified without mocking sys.exit to avoid UnboundLocalError)."""
    malformed_xml = "this is not <xml>"
    
    with patch("scripts.download_papers.make_request_with_retry") as mock_make_request:
        mock_make_request.return_value = malformed_xml.encode('utf-8')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "sources")
            registry_path = os.path.join(output_dir, "registry.json")
            
            with pytest.raises(RuntimeError) as excinfo:
                download_arxiv_papers(
                    query="all:AI",
                    max_results=2,
                    output_dir=output_dir,
                    registry_path=registry_path
                )
            assert "Critical error fetching metadata" in str(excinfo.value)

@patch("scripts.download_papers.make_request_with_retry")
def test_stress_missing_fields_in_feed(mock_make_request):
    """Verify that download_papers skips entries with missing titles or missing PDF links, but handles missing IDs using a fallback."""
    # Entry 1: Missing Title
    # Entry 2: Missing PDF Link
    # Entry 3: Missing ID (should fallback to PDF link)
    # Entry 4: Valid
    xml_feed = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00001" />
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2401.00002v1</id>
        <title>Missing PDF Link</title>
      </entry>
      <entry>
        <title>Missing ID but has PDF link</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00003" />
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2401.00004v1</id>
        <title>Valid Paper</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00004" />
      </entry>
    </feed>
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "sources")
        registry_path = os.path.join(output_dir, "registry.json")
        
        mock_make_request.side_effect = [
            xml_feed.encode('utf-8'), # API XML feed
            True,                     # Download for Entry 3 (fallback ID)
            True                      # Download for Entry 4 (valid)
        ]
        
        download_arxiv_papers(
            query="all:AI",
            max_results=4,
            output_dir=output_dir,
            registry_path=registry_path
        )
        
        # Entry 3 fallback ID is extracted from http://export.arxiv.org/pdf/2401.00003
        # Which is "2401.00003".
        # Let's check registry to see if both are registered
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
            
        assert "2401.00003" in registry
        assert "2401.00004" in registry
        assert len(registry) == 2


# ---------------------------------------------------------
# Merged tests from stress_test_ingestion.py
# ---------------------------------------------------------

# 1. Test empty queries
@patch("scripts.download_papers.make_request_with_retry")
def test_empty_query_arxiv_behavior(mock_make_request, tmp_path):
    """Verify behavior of download_arxiv_papers with empty query."""
    # Simulate empty search results (empty feed)
    empty_xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
    </feed>"""
    mock_make_request.return_value = empty_xml.encode('utf-8')
    
    output_dir = tmp_path / "dummy_out"
    registry_path = tmp_path / "dummy_reg.json"
    
    res = download_arxiv_papers(
        query="",
        max_results=5,
        output_dir=str(output_dir),
        registry_path=str(registry_path)
    )
    # Should complete successfully and return the registry dict
    assert isinstance(res, dict)

# 2. Test invalid directories & retry on OSError
@patch("os.makedirs")
@patch("scripts.download_papers.make_request_with_retry")
def test_invalid_directory_creation_raises(mock_make_request, mock_makedirs, tmp_path):
    """Verify that download_arxiv_papers raises OSError when directory cannot be created."""
    mock_makedirs.side_effect = PermissionError("Permission denied")
    registry_path = tmp_path / "dummy_reg.json"
    
    with pytest.raises(PermissionError):
        download_arxiv_papers(
            query="all:AI",
            max_results=5,
            output_dir="/invalid/dir/path",
            registry_path=str(registry_path)
        )

@patch("time.sleep", return_value=None)
@patch("urllib.request.urlretrieve")
def test_make_request_with_retry_does_not_retry_disk_errors(mock_urlretrieve, mock_sleep, tmp_path):
    """Verify that make_request_with_retry does not retry local disk errors."""
    # Simulate local disk PermissionError on writing PDF
    mock_urlretrieve.side_effect = PermissionError("Disk not writable")
    filepath = tmp_path / "dummy.pdf"
    
    with pytest.raises(PermissionError):
        make_request_with_retry("http://example.com/pdf", is_download=True, filepath=str(filepath))
        
    # It should not retry disk errors, so call count should be 1
    assert mock_urlretrieve.call_count == 1
    mock_sleep.assert_not_called()

# 3. Test socket timeouts
@patch("time.sleep", return_value=None)
@patch("urllib.request.urlopen")
def test_socket_timeout_retry(mock_urlopen, mock_sleep):
    """Verify that socket timeouts are retried 3 times and then re-raised."""
    mock_urlopen.side_effect = TimeoutError("Connection timed out")
    
    with pytest.raises(TimeoutError):
        make_request_with_retry("http://example.com/api", is_download=False)
        
    assert mock_urlopen.call_count == 3

# 4. Test rate limits (3s sleep) and failure bypass
@patch("urllib.request.urlretrieve")
@patch("os.replace")
def test_rate_limit_bypass_on_failure(mock_replace, mock_urlretrieve, tmp_path):
    """Verify that the 3-second sleep is bypassed on failure, violating the rate limit."""
    # We will track sleep calls
    sleep_calls = []
    def custom_sleep(sec):
        sleep_calls.append(sec)
        
    # Simulate a network failure on downloading PDF (raises exception)
    mock_urlretrieve.side_effect = urllib.error.URLError("Connection refused")
    filepath = tmp_path / "dummy.pdf"
    
    with patch("time.sleep", side_effect=custom_sleep):
        try:
            make_request_with_retry("http://example.com/pdf", is_download=True, filepath=str(filepath))
        except Exception:
            pass
            
    # Check the sleep calls.
    # In make_request_with_retry, the backoff sleep is called for attempts 1 and 2:
    # Attempt 1 fails: backoff = 3 * (2**0) = 3s
    # Attempt 2 fails: backoff = 3 * (2**1) = 6s
    # Attempt 3 fails: raises exception, no success sleep (3s) is called.
    # Total sleep calls during retry loop: [3, 6]
    assert sleep_calls == [3, 6]

# 5. Test invalid arXiv responses
@patch("scripts.download_papers.make_request_with_retry")
def test_invalid_arxiv_xml_handling(mock_make_request, tmp_path):
    """Verify that malformed XML is handled and script exits with code 1."""
    malformed_xml = "<feed><entry><title>Malformed XML" # Missing closing tags
    mock_make_request.return_value = malformed_xml.encode('utf-8')
    output_dir = tmp_path / "dummy_out"
    registry_path = tmp_path / "dummy_reg.json"
    
    with pytest.raises(RuntimeError) as excinfo:
        download_arxiv_papers(
            query="all:AI",
            max_results=5,
            output_dir=str(output_dir),
            registry_path=str(registry_path)
        )
    assert "Critical error fetching metadata" in str(excinfo.value)

@patch("scripts.download_papers.make_request_with_retry")
def test_missing_xml_fields(mock_make_request, tmp_path):
    """Verify that entries missing titles or pdf links are handled safely."""
    # 1st entry has missing title
    # 2nd entry has missing pdf link
    # 3rd entry is valid
    xml_data = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00001" />
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2401.00002v1</id>
        <title>Missing PDF Link</title>
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2401.00003v1</id>
        <title>Valid Entry</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00003" />
      </entry>
    </feed>
    """
    # XML query, then PDF download
    mock_make_request.side_effect = [xml_data.encode('utf-8'), True]
    output_dir = tmp_path / "dummy_out"
    registry_path = tmp_path / "dummy_reg.json"
    
    res = download_arxiv_papers(
        query="all:AI",
        max_results=3,
        output_dir=str(output_dir),
        registry_path=str(registry_path)
    )
    # The valid entry should be downloaded
    mock_make_request.assert_any_call("http://export.arxiv.org/pdf/2401.00003.pdf", is_download=True, filepath=os.path.join(str(output_dir), "Valid Entry.pdf"))
    # Check that it completed successfully
    assert isinstance(res, dict)
