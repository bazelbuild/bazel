// doc generates HTML files from the comments in header files.
//
// doc expects to be given the path to a JSON file via the --config option.
// From that JSON (which is defined by the Config struct) it reads a list of
// header file locations and generates HTML files for each in the current
// directory.

package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

// Config describes the structure of the config JSON file.
type Config struct {
	// BaseDirectory is a path to which other paths in the file are
	// relative.
	BaseDirectory string
	Sections      []ConfigSection
}

type ConfigSection struct {
	Name string
	// Headers is a list of paths to header files.
	Headers []string
}

// HeaderFile is the internal representation of a header file.
type HeaderFile struct {
	// Name is the basename of the header file (e.g. "ex_data.html").
	Name string
	// Preamble contains a comment for the file as a whole. Each string
	// is a separate paragraph.
	Preamble []string
	Sections []HeaderSection
}

type HeaderSection struct {
	// Preamble contains a comment for a group of functions.
	Preamble []string
	Decls    []HeaderDecl
	// Anchor, if non-empty, is the URL fragment to use in anchor tags.
	Anchor string
	// IsPrivate is true if the section contains private functions (as
	// indicated by its name).
	IsPrivate bool
}

type HeaderDecl struct {
	// Comment contains a comment for a specific function. Each string is a
	// paragraph. Some paragraph may contain \n runes to indicate that they
	// are preformatted.
	Comment []string
	// Name contains the name of the function, if it could be extracted.
	Name string
	// Decl contains the preformatted C declaration itself.
	Decl string
	// Anchor, if non-empty, is the URL fragment to use in anchor tags.
	Anchor string
}

const (
	cppGuard     = "#if defined(__cplusplus)"
	commentStart = "/* "
	commentEnd   = " */"
)

func extractComment(lines []string, lineNo int) (comment []string, rest []string, restLineNo int, err error) {
	if len(lines) == 0 {
		return nil, lines, lineNo, nil
	}

	restLineNo = lineNo
	rest = lines

	if !strings.HasPrefix(rest[0], commentStart) {
		panic("extractComment called on non-comment")
	}
	commentParagraph := rest[0][len(commentStart):]
	rest = rest[1:]
	restLineNo++

	for len(rest) > 0 {
		i := strings.Index(commentParagraph, commentEnd)
		if i >= 0 {
			if i != len(commentParagraph)-len(commentEnd) {
				err = fmt.Errorf("garbage after comment end on line %d", restLineNo)
				return
			}
			commentParagraph = commentParagraph[:i]
			if len(commentParagraph) > 0 {
				comment = append(comment, commentParagraph)
			}
			return
		}

		line := rest[0]
		if !strings.HasPrefix(line, " *") {
			err = fmt.Errorf("comment doesn't start with block prefix on line %d: %s", restLineNo, line)
			return
		}
		if len(line) == 2 || line[2] != '/' {
			line = line[2:]
		}
		if strings.HasPrefix(line, "   ") {
			/* Identing the lines of a paragraph marks them as
			* preformatted. */
			if len(commentParagraph) > 0 {
				commentParagraph += "\n"
			}
			line = line[3:]
		}
		if len(line) > 0 {
			commentParagraph = commentParagraph + line
			if len(commentParagraph) > 0 && commentParagraph[0] == ' ' {
				commentParagraph = commentParagraph[1:]
			}
		} else {
			comment = append(comment, commentParagraph)
			commentParagraph = ""
		}
		rest = rest[1:]
		restLineNo++
	}

	err = errors.New("hit EOF in comment")
	return
}

func extractDecl(lines []string, lineNo int) (decl string, rest []string, restLineNo int, err error) {
	if len(lines) == 0 {
		return "", lines, lineNo, nil
	}

	rest = lines
	restLineNo = lineNo

	var stack []rune
	for len(rest) > 0 {
		line := rest[0]
		for _, c := range line {
			switch c {
			case '(', '{', '[':
				stack = append(stack, c)
			case ')', '}', ']':
				if len(stack) == 0 {
					err = fmt.Errorf("unexpected %c on line %d", c, restLineNo)
					return
				}
				var expected rune
				switch c {
				case ')':
					expected = '('
				case '}':
					expected = '{'
				case ']':
					expected = '['
				default:
					panic("internal error")
				}
				if last := stack[len(stack)-1]; last != expected {
					err = fmt.Errorf("found %c when expecting %c on line %d", c, last, restLineNo)
					return
				}
				stack = stack[:len(stack)-1]
			}
		}
		if len(decl) > 0 {
			decl += "\n"
		}
		decl += line
		rest = rest[1:]
		restLineNo++

		if len(stack) == 0 && (len(decl) == 0 || decl[len(decl)-1] != '\\') {
			break
		}
	}

	return
}

func skipLine(s string) string {
	i := strings.Index(s, "\n")
	if i > 0 {
		return s[i:]
	}
	return ""
}

func getNameFromDecl(decl string) (string, bool) {
	for strings.HasPrefix(decl, "#if") || strings.HasPrefix(decl, "#elif") {
		decl = skipLine(decl)
	}
	if strings.HasPrefix(decl, "struct ") {
		return "", false
	}
	if strings.HasPrefix(decl, "#define ") {
		// This is a preprocessor #define. The name is the next symbol.
		decl = strings.TrimPrefix(decl, "#define ")
		for len(decl) > 0 && decl[0] == ' ' {
			decl = decl[1:]
		}
		i := strings.IndexAny(decl, "( ")
		if i < 0 {
			return "", false
		}
		return decl[:i], true
	}
	decl = strings.TrimPrefix(decl, "OPENSSL_EXPORT ")
	decl = strings.TrimPrefix(decl, "STACK_OF(")
	decl = strings.TrimPrefix(decl, "LHASH_OF(")
	i := strings.Index(decl, "(")
	if i < 0 {
		return "", false
	}
	j := strings.LastIndex(decl[:i], " ")
	if j < 0 {
		return "", false
	}
	for j+1 < len(decl) && decl[j+1] == '*' {
		j++
	}
	return decl[j+1 : i], true
}

func sanitizeAnchor(name string) string {
	return strings.Replace(name, " ", "-", -1)
}

func (config *Config) parseHeader(path string) (*HeaderFile, error) {
	headerPath := filepath.Join(config.BaseDirectory, path)

	headerFile, err := os.Open(headerPath)
	if err != nil {
		return nil, err
	}
	defer headerFile.Close()

	scanner := bufio.NewScanner(headerFile)
	var lines, oldLines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	lineNo := 0
	found := false
	for i, line := range lines {
		lineNo++
		if line == cppGuard {
			lines = lines[i+1:]
			lineNo++
			found = true
			break
		}
	}

	if !found {
		return nil, errors.New("no C++ guard found")
	}

	if len(lines) == 0 || lines[0] != "extern \"C\" {" {
		return nil, errors.New("no extern \"C\" found after C++ guard")
	}
	lineNo += 2
	lines = lines[2:]

	header := &HeaderFile{
		Name: filepath.Base(path),
	}

	for i, line := range lines {
		lineNo++
		if len(line) > 0 {
			lines = lines[i:]
			break
		}
	}

	oldLines = lines
	if len(lines) > 0 && strings.HasPrefix(lines[0], commentStart) {
		comment, rest, restLineNo, err := extractComment(lines, lineNo)
		if err != nil {
			return nil, err
		}

		if len(rest) > 0 && len(rest[0]) == 0 {
			if len(rest) < 2 || len(rest[1]) != 0 {
				return nil, errors.New("preamble comment should be followed by two blank lines")
			}
			header.Preamble = comment
			lineNo = restLineNo + 2
			lines = rest[2:]
		} else {
			lines = oldLines
		}
	}

	allAnchors := make(map[string]struct{})

	for {
		// Start of a section.
		if len(lines) == 0 {
			return nil, errors.New("unexpected end of file")
		}
		line := lines[0]
		if line == cppGuard {
			break
		}

		if len(line) == 0 {
			return nil, fmt.Errorf("blank line at start of section on line %d", lineNo)
		}

		var section HeaderSection

		if strings.HasPrefix(line, commentStart) {
			comment, rest, restLineNo, err := extractComment(lines, lineNo)
			if err != nil {
				return nil, err
			}
			if len(rest) > 0 && len(rest[0]) == 0 {
				anchor := sanitizeAnchor(firstSentence(comment))
				if len(anchor) > 0 {
					if _, ok := allAnchors[anchor]; ok {
						return nil, fmt.Errorf("duplicate anchor: %s", anchor)
					}
					allAnchors[anchor] = struct{}{}
				}

				section.Preamble = comment
				section.IsPrivate = len(comment) > 0 && strings.HasPrefix(comment[0], "Private functions")
				section.Anchor = anchor
				lines = rest[1:]
				lineNo = restLineNo + 1
			}
		}

		for len(lines) > 0 {
			line := lines[0]
			if len(line) == 0 {
				lines = lines[1:]
				lineNo++
				break
			}
			if line == cppGuard {
				return nil, errors.New("hit ending C++ guard while in section")
			}

			var comment []string
			var decl string
			if strings.HasPrefix(line, commentStart) {
				comment, lines, lineNo, err = extractComment(lines, lineNo)
				if err != nil {
					return nil, err
				}
			}
			if len(lines) == 0 {
				return nil, errors.New("expected decl at EOF")
			}
			decl, lines, lineNo, err = extractDecl(lines, lineNo)
			if err != nil {
				return nil, err
			}
			name, ok := getNameFromDecl(decl)
			if !ok {
				name = ""
			}
			if last := len(section.Decls) - 1; len(name) == 0 && len(comment) == 0 && last >= 0 {
				section.Decls[last].Decl += "\n" + decl
			} else {
				anchor := sanitizeAnchor(name)
				// TODO(davidben): Enforce uniqueness. This is
				// skipped because #ifdefs currently result in
				// duplicate table-of-contents entries.
				allAnchors[anchor] = struct{}{}

				section.Decls = append(section.Decls, HeaderDecl{
					Comment: comment,
					Name:    name,
					Decl:    decl,
					Anchor:  anchor,
				})
			}

			if len(lines) > 0 && len(lines[0]) == 0 {
				lines = lines[1:]
				lineNo++
			}
		}

		header.Sections = append(header.Sections, section)
	}

	return header, nil
}

func firstSentence(paragraphs []string) string {
	if len(paragraphs) == 0 {
		return ""
	}
	s := paragraphs[0]
	i := strings.Index(s, ". ")
	if i >= 0 {
		return s[:i]
	}
	if lastIndex := len(s) - 1; s[lastIndex] == '.' {
		return s[:lastIndex]
	}
	return s
}

func markupPipeWords(s string) template.HTML {
	ret := ""

	for {
		i := strings.Index(s, "|")
		if i == -1 {
			ret += s
			break
		}
		ret += s[:i]
		s = s[i+1:]

		i = strings.Index(s, "|")
		j := strings.Index(s, " ")
		if i > 0 && (j == -1 || j > i) {
			ret += "<tt>"
			ret += s[:i]
			ret += "</tt>"
			s = s[i+1:]
		} else {
			ret += "|"
		}
	}

	return template.HTML(ret)
}

func markupFirstWord(s template.HTML) template.HTML {
	start := 0
again:
	end := strings.Index(string(s[start:]), " ")
	if end > 0 {
		end += start
		w := strings.ToLower(string(s[start:end]))
		if w == "a" || w == "an" {
			start = end + 1
			goto again
		}
		return s[:start] + "<span class=\"first-word\">" + s[start:end] + "</span>" + s[end:]
	}
	return s
}

func newlinesToBR(html template.HTML) template.HTML {
	s := string(html)
	if !strings.Contains(s, "\n") {
		return html
	}
	s = strings.Replace(s, "\n", "<br>", -1)
	s = strings.Replace(s, " ", "&nbsp;", -1)
	return template.HTML(s)
}

func generate(outPath string, config *Config) (map[string]string, error) {
	headerTmpl := template.New("headerTmpl")
	headerTmpl.Funcs(template.FuncMap{
		"firstSentence":   firstSentence,
		"markupPipeWords": markupPipeWords,
		"markupFirstWord": markupFirstWord,
		"newlinesToBR":    newlinesToBR,
	})
	headerTmpl, err := headerTmpl.Parse(`<!DOCTYPE html>
<html>
  <head>
    <title>BoringSSL - {{.Name}}</title>
    <meta charset="utf-8">
    <link rel="stylesheet" type="text/css" href="doc.css">
  </head>

  <body>
    <div id="main">
    <h2>{{.Name}}</h2>

    {{range .Preamble}}<p>{{. | html | markupPipeWords}}</p>{{end}}

    <ol>
      {{range .Sections}}
        {{if not .IsPrivate}}
          {{if .Anchor}}<li class="header"><a href="#{{.Anchor}}">{{.Preamble | firstSentence | html | markupPipeWords}}</a></li>{{end}}
          {{range .Decls}}
            {{if .Anchor}}<li><a href="#{{.Anchor}}"><tt>{{.Name}}</tt></a></li>{{end}}
          {{end}}
        {{end}}
      {{end}}
    </ol>

    {{range .Sections}}
      {{if not .IsPrivate}}
        <div class="section">
        {{if .Preamble}}
          <div class="sectionpreamble">
          <a{{if .Anchor}} name="{{.Anchor}}"{{end}}>
          {{range .Preamble}}<p>{{. | html | markupPipeWords}}</p>{{end}}
          </a>
          </div>
        {{end}}

        {{range .Decls}}
          <div class="decl">
          <a{{if .Anchor}} name="{{.Anchor}}"{{end}}>
          {{range .Comment}}
            <p>{{. | html | markupPipeWords | newlinesToBR | markupFirstWord}}</p>
          {{end}}
          <pre>{{.Decl}}</pre>
          </a>
          </div>
        {{end}}
        </div>
      {{end}}
    {{end}}
    </div>
  </body>
</html>`)
	if err != nil {
		return nil, err
	}

	headerDescriptions := make(map[string]string)

	for _, section := range config.Sections {
		for _, headerPath := range section.Headers {
			header, err := config.parseHeader(headerPath)
			if err != nil {
				return nil, errors.New("while parsing " + headerPath + ": " + err.Error())
			}
			headerDescriptions[header.Name] = firstSentence(header.Preamble)
			filename := filepath.Join(outPath, header.Name+".html")
			file, err := os.OpenFile(filename, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
			if err != nil {
				panic(err)
			}
			defer file.Close()
			if err := headerTmpl.Execute(file, header); err != nil {
				return nil, err
			}
		}
	}

	return headerDescriptions, nil
}

func generateIndex(outPath string, config *Config, headerDescriptions map[string]string) error {
	indexTmpl := template.New("indexTmpl")
	indexTmpl.Funcs(template.FuncMap{
		"baseName": filepath.Base,
		"headerDescription": func(header string) string {
			return headerDescriptions[header]
		},
	})
	indexTmpl, err := indexTmpl.Parse(`<!DOCTYPE html5>

  <head>
    <title>BoringSSL - Headers</title>
    <meta charset="utf-8">
    <link rel="stylesheet" type="text/css" href="doc.css">
  </head>

  <body>
    <div id="main">
      <table>
        {{range .Sections}}
	  <tr class="header"><td colspan="2">{{.Name}}</td></tr>
	  {{range .Headers}}
	    <tr><td><a href="{{. | baseName}}.html">{{. | baseName}}</a></td><td>{{. | baseName | headerDescription}}</td></tr>
	  {{end}}
	{{end}}
      </table>
    </div>
  </body>
</html>`)

	if err != nil {
		return err
	}

	file, err := os.OpenFile(filepath.Join(outPath, "headers.html"), os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	if err := indexTmpl.Execute(file, config); err != nil {
		return err
	}

	return nil
}

func copyFile(outPath string, inFilePath string) error {
	bytes, err := ioutil.ReadFile(inFilePath)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(filepath.Join(outPath, filepath.Base(inFilePath)), bytes, 0666)
}

func main() {
	var (
		configFlag *string = flag.String("config", "doc.config", "Location of config file")
		outputDir  *string = flag.String("out", ".", "Path to the directory where the output will be written")
		config     Config
	)

	flag.Parse()

	if len(*configFlag) == 0 {
		fmt.Printf("No config file given by --config\n")
		os.Exit(1)
	}

	if len(*outputDir) == 0 {
		fmt.Printf("No output directory given by --out\n")
		os.Exit(1)
	}

	configBytes, err := ioutil.ReadFile(*configFlag)
	if err != nil {
		fmt.Printf("Failed to open config file: %s\n", err)
		os.Exit(1)
	}

	if err := json.Unmarshal(configBytes, &config); err != nil {
		fmt.Printf("Failed to parse config file: %s\n", err)
		os.Exit(1)
	}

	headerDescriptions, err := generate(*outputDir, &config)
	if err != nil {
		fmt.Printf("Failed to generate output: %s\n", err)
		os.Exit(1)
	}

	if err := generateIndex(*outputDir, &config, headerDescriptions); err != nil {
		fmt.Printf("Failed to generate index: %s\n", err)
		os.Exit(1)
	}

	if err := copyFile(*outputDir, "doc.css"); err != nil {
		fmt.Printf("Failed to copy static file: %s\n", err)
		os.Exit(1)
	}
}
