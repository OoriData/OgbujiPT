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
    Split string and generate the sequence of chunks

    >>> from ogbujipt.html_helper import html_split
    >>> list(text_split('She sells seashells by the seashore', chunk_size=5, separator=' '))
    ['She', 'sells', 'seashells', 'by', 'the', 'seashore']
    >>> list(text_split('She sells seashells by the seashore', chunk_size=10, separator=' '))
    ['She sells', 'seashells', 'by the', 'seashore']
    # Notice case sensitivity, plus the fact that the separator is not included in the chunks
    >>> list(text_split('She sells seashells by the seashore', chunk_size=10, separator='s'))
    ['She ', 'ells ', 'ea', 'hell', ' by the ', 'ea', 'hore']
    >>> list(text_split('She\tsells seashells\tby the seashore', chunk_size=10, separator='\\s'))
    ['She\tsells', 'seashells', 'by the', 'seashore']
    >>> list(text_split('She\tsells seashells\tby the seashore', chunk_size=10, separator='\\s', joiner=' '))
    ['She sells', 'seashells', 'by the', 'seashore']

    Args:
        html (str): HTML string to be split into chunks

        chunk_size (int): Guidance on maximum length (based on distance_function) of each chunk

        separator (str, optional): Regex used to split `text` into sections. Do not include outer capturing parenthesis.
            Don't forget to use escaping where necessary.

        joiner (str, optional): Exact string used to rejoin any sections in order to meet target length
            defaults to using the literal match from the separator

        len_func (callable, optional): Function to measure chunk length, len() by default

    Returns:
        chunks (List[str]): List of chunks of the text provided
    
    TODO: Implementation pending
    '''
    raise NotImplementedError("html_split is not yet implemented")


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
    Convert a single element to markdown and append to chunks list
    
    TODO: Implementation pending
    '''
    raise NotImplementedError("elem2markdown is not yet implemented")

def html2markdown(root, chunks):
    '''
    Converts HTML text or a selectolax parsed tree to Markdown, using probably oversimplified logic
    Preserves indentation in code blocks as does the inspiring code:
    https://github.com/SivilTaram/code-html-to-markdown/blob/main/main.py
    
    Args:
        root: selectolax HTMLParser tree or root element
        chunks: list to append markdown chunks to
    
    Returns:
        markdown string (or appends to chunks if chunks is provided)
    
    TODO: Implementation pending - function body contains incomplete/incorrect code
    '''
    raise NotImplementedError("html2markdown is not yet implemented")


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
