// Package ui holds the browser front end, embedded into the binary.
package ui

import (
	"embed"
	"io/fs"
)

//go:embed static
var files embed.FS

// FS is the web root (index.html, app.css, app.js).
func FS() fs.FS {
	sub, err := fs.Sub(files, "static")
	if err != nil {
		panic(err)
	}
	return sub
}
