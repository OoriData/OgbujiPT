# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.html_helper

'''
HTML Chunking: first of all peprocesses HTML to remove hidden content, then converts to Markdown,
which is chunked with some semantic awareness

Some code adapted from https://github.com/KLGR123/html_chunking/blob/main/html_chunking.py
'''
import re
import warnings

# Just use tiktoken, even though it's OpenAI specific
import tiktoken

try:
    import cssutils  # pip install cssutils
except ImportError:
    cssutils = None
    warnings.warn(
        'Using html_helper without cssutils, which will limit features. May want to do e.g. `pip install cssutils`')

# selectolax install is optional for OgbujiPT, but mandatory for html_helper
try:
    from selectolax.parser import HTMLParser  # pip install selectolax
except ImportError:
    HTMLParser = None
    warnings.warn(
        'Cannot use html_helper without selectolax. May want to do e.g. `pip install selectolax`')



HTML_SAMPLE1 = '''\
<!DOCTYPE html>
<html>
<head>
<style>
h1.hidden {
  display: none;
}
</style>
</head>
<body>

<h1>Here's is a visible heading</h1>
<h1 class="hidden">Here's is a hidden heading</h1>
<p>The quick brown fox jumps over the lazy dog. <a href='http://example.org'>Learn More</a>.</p>

</body>
</html>
'''

HTML_SAMPLE2 = '''\
<!DOCTYPE html>
<html>
<head>
<style>
#page1 {
    display: block; /* shown by default */
}

#page2, #page3, #page4, #page5 {
    display: none;
}
</style>
</head>
<body>

    <div class="content">
        <div class="page" id="page1">
            <p> Page 1 text goes here</p>
        </div>
        <div class="page" id="page2">
            <p> Page 2 text goes here</p>
        </div>
    </div>

</body>
</html>
'''


HTML_ANTIPATTERN1 = '''\
        <div class="content">
            <div class="page" id="page1" style="display: block">
                <p> Page 1 text goes here</p>
            </div>
            <div class="page" id="page2" style="display: none">
                <p> Page 2 text goes here</p>
            </div>
'''


def html_split(text: str, chunk_size: int, separator: str='\n\n', joiner=None, len_func=len):
    '''
    Split string and generate the sequence of chunks.
    
    This is a wrapper around text_split from text_helper, intended for use with markdown
    text that has been converted from HTML.

    >>> from ogbujipt.text.html import html_split
    >>> list(html_split('She sells seashells by the seashore', chunk_size=5, separator=' '))
    ['She', 'sells', 'seashells', 'by', 'the', 'seashore']
    >>> list(html_split('She sells seashells by the seashore', chunk_size=10, separator=' '))
    ['She sells', 'seashells', 'by the', 'seashore']

    Args:
        text (str): Text string (typically markdown) to be split into chunks

        chunk_size (int): Guidance on maximum length (based on len_func) of each chunk

        separator (str, optional): Regex used to split `text` into sections. Do not include outer capturing parenthesis.
            Don't forget to use escaping where necessary.

        joiner (str, optional): Exact string used to rejoin any sections in order to meet target length
            defaults to using the literal match from the separator

        len_func (callable, optional): Function to measure chunk length, len() by default

    Yields:
        chunks (str): Chunks of the text provided
    '''
    from ogbujipt.text.splitter import text_split
    yield from text_split(text, chunk_size=chunk_size, separator=separator, joiner=joiner, len_func=len_func)


def count_tokens(text: str) -> int:
    encoder = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = encoder.encode(text)
    return len(tokens)


def check_hidden_by_local_style(elem):
    '''Check whether element should be hidden based on its own style attrs'''
    css_text = elem.attributes.get('style')
    if css_text:
        sheet = cssutils.parseString(css_text)
        for rule in sheet:
            if rule.type == rule.STYLE_RULE:
                if 'display' in rule.style and 'none' in rule.style['display']:
                    return True
                elif 'visibility' in rule.style and 'hidden' in rule.style['visibility']:
                    return True
    return False

ATTRS_TO_TRIM = ['href', 'src', 'd', 'url', 'data-url', 'data-src', 'data-src-hq']


def clean_html(html: str, attr_max_len: int = 0) -> tuple[str, str]:
    '''
    Remove text which would be hidden according to styling rules
    '''
    tree = HTMLParser(html)
    removed_content = []  # Records content in elements removed according to style rules

    css_texts = [style.text(deep=True, separator='', strip=False) for style in tree.css('style')]
    for css_text in css_texts:
        sheet = cssutils.parseString(css_text)
        for rule in sheet:
            if rule.type == rule.STYLE_RULE:
                selector = rule.selectorText
                if '::' in selector or ':after' in selector or ':before' in selector:
                    continue
                if 'display' in rule.style and 'none' in rule.style['display']:
                    for element in tree.css(selector):
                        removed_content.append(element.text(deep=True, separator='', strip=False))
                        element.decompose()
                elif 'visibility' in rule.style and 'hidden' in rule.style['visibility']:
                    for element in tree.css(selector):
                        removed_content.append(element.text(deep=True, separator='', strip=False))
                        element.decompose()

    for tag in ['script', 'style']:
        for element in tree.css(tag):
            removed_content.append(element.text(deep=True, separator='', strip=False))
            element.decompose()

    for elem in tree.css('*'):
        if check_hidden_by_local_style(elem):
            removed_content.append(elem.text(deep=True, separator='', strip=False))
            elem.decompose()

        if elem.attrs.get('aria-hidden') == 'true' or elem.attrs.get('tabindex') == '-1':
            removed_content.append(elem.text(deep=True, separator='', strip=False))
            elem.decompose()

    if attr_max_len:
        for elem in tree.css('*'):
            for attr in ATTRS_TO_TRIM:
                if attr in elem.attrs:
                    if len(elem.attrs[attr]) > attr_max_len:
                        elem.attrs[attr] = elem.attrs[attr][:attr_max_len] + "…"

    return tree, removed_content


def elem2markdown(elem, chunks):
    '''
    Convert a single HTML element to markdown and append to chunks list.
    
    This is a helper function used by html2markdown for recursive processing.
    Note: This function is kept for backward compatibility but html2markdown
    now uses _process_element_to_markdown internally.
    
    Args:
        elem: selectolax element node
        chunks: list to append markdown strings to
    '''
    # Delegate to the main processing function
    _process_element_to_markdown(elem, chunks)

def html2markdown(root, chunks=None):
    '''
    Converts HTML text or a selectolax parsed tree to Markdown, using simplified logic.
    Preserves indentation in code blocks as does the inspiring code:
    https://github.com/SivilTaram/code-html-to-markdown/blob/main/main.py
    
    Args:
        root: selectolax HTMLParser tree, root element, or HTML string
        chunks: optional list to append markdown chunks to. If None, returns a string.
    
    Returns:
        If chunks is provided, appends to chunks and returns None.
        Otherwise, returns markdown string.
    '''
    if HTMLParser is None:
        raise ImportError("selectolax is required for html2markdown")
    
    # If root is a string, parse it
    if isinstance(root, str):
        root = HTMLParser(root)
    
    # Get body or use root
    body = root.body if hasattr(root, 'body') and root.body else root
    
    if chunks is None:
        chunks = []
        return_string = True
    else:
        return_string = False
    
    # Process the body element
    if body:
        # Get title if available
        title_elem = root.css_first('title')
        if title_elem:
            title_text = title_elem.text(deep=True, separator='', strip=True)
            if title_text:
                chunks.append(f"# {title_text}\n\n")
        
        # Process main content
        _process_element_to_markdown(body, chunks)
    
    if return_string:
        result = ''.join(chunks)
        # Apply post-processing
        result = post_processing(result)
        return result
    else:
        # Post-process chunks in place
        if chunks:
            combined = ''.join(chunks)
            processed = post_processing(combined)
            chunks.clear()
            chunks.append(processed)


def _process_element_to_markdown(elem, chunks, level=0):
    '''
    Recursively process an HTML element and convert to markdown.
    
    Args:
        elem: selectolax element node
        chunks: list to append markdown strings to
        level: nesting level (for indentation)
    '''
    if not elem or not hasattr(elem, 'tag'):
        return
    
    tag = elem.tag.lower() if elem.tag else ''
    text = elem.text(deep=False, separator='', strip=True) if hasattr(elem, 'text') else ''
    
    # Handle different HTML tags
    if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        level_num = int(tag[1])
        prefix = '#' * level_num + ' '
        if text:
            chunks.append(f"\n{prefix}{text}\n\n")
    elif tag == 'p':
        if text:
            chunks.append(f"{text}\n\n")
    elif tag == 'strong' or tag == 'b':
        if text:
            chunks.append(f"**{text}**")
    elif tag == 'em' or tag == 'i':
        if text:
            chunks.append(f"*{text}*")
    elif tag == 'code':
        # Check if parent is pre for code blocks
        parent_tag = elem.parent.tag.lower() if elem.parent and hasattr(elem.parent, 'tag') else ''
        if parent_tag == 'pre':
            # Handled by pre tag
            pass
        else:
            if text:
                chunks.append(f"`{text}`")
    elif tag == 'pre':
        code_text = elem.text(deep=True, separator='\n', strip=False)
        if code_text:
            chunks.append(f"\n```\n{code_text}\n```\n\n")
        return  # Don't process children for pre
    elif tag == 'a':
        href = elem.attrs.get('href', '') if hasattr(elem, 'attrs') else ''
        link_text = elem.text(deep=True, separator='', strip=True)
        if link_text:
            if href:
                chunks.append(f"[{link_text}]({href})")
            else:
                chunks.append(link_text)
    elif tag == 'ul':
        chunks.append('\n')
        for child in elem.iter():
            if child != elem and hasattr(child, 'tag') and child.tag.lower() == 'li':
                _process_element_to_markdown(child, chunks, level + 1)
        chunks.append('\n')
        return
    elif tag == 'ol':
        chunks.append('\n')
        idx = 1
        for child in elem.iter():
            if child != elem and hasattr(child, 'tag') and child.tag.lower() == 'li':
                li_text = child.text(deep=True, separator='', strip=True)
                if li_text:
                    chunks.append(f"{idx}. {li_text}\n")
                idx += 1
        chunks.append('\n')
        return
    elif tag == 'li':
        li_text = elem.text(deep=True, separator='', strip=True)
        if li_text:
            chunks.append(f"* {li_text}\n")
    elif tag == 'blockquote':
        quote_text = elem.text(deep=True, separator='\n', strip=True)
        if quote_text:
            lines = quote_text.split('\n')
            for line in lines:
                chunks.append(f"> {line}\n")
            chunks.append('\n')
    elif tag == 'table':
        # Simple table conversion
        rows = elem.css('tr')
        if rows:
            # Header row
            header_cells = rows[0].css('th, td')
            if header_cells:
                header_texts = [cell.text(deep=True, separator='', strip=True) for cell in header_cells]
                chunks.append('\n| ' + ' | '.join(header_texts) + ' |\n')
                chunks.append('| ' + ' | '.join(['---'] * len(header_texts)) + ' |\n')
            
            # Data rows
            for row in rows[1:]:
                cells = row.css('td')
                if cells:
                    cell_texts = [cell.text(deep=True, separator='', strip=True) for cell in cells]
                    chunks.append('| ' + ' | '.join(cell_texts) + ' |\n')
            chunks.append('\n')
        return
    elif tag in ['div', 'span', 'section', 'article', 'main', 'body']:
        # Container elements - just process children, add spacing for div/section/article
        if tag in ['div', 'section', 'article'] and level == 0:
            chunks.append('\n')
    elif tag in ['script', 'style', 'head', 'meta']:
        # Skip these
        return
    else:
        # For unknown tags, just extract text
        if text:
            chunks.append(text)
    
    # Process children recursively (unless we returned early)
    for child in elem.iter():
        if child != elem:
            _process_element_to_markdown(child, chunks, level + 1)


def post_processing(markdown_content: str):
    """
    Post processing to remove extra spaces and new lines.
    """
    lines = [line for line in markdown_content.split("\n")]
    # if this line matches spaces, then replace with ""
    lines = [re.sub(r"^\s+$", "", line) for line in lines]
    markdown_content = "\n".join(lines)
    markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
    # replace ` .` with `.
    markdown_content = re.sub(r"` \.", "`.", markdown_content)
    markdown_content = re.sub(r"` \,", "`,", markdown_content)
    return markdown_content.strip()
