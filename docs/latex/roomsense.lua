-- Pandoc filter for the RoomSense PDFs (used by docs/build_pdfs.py).
--  * drops the Markdown title (the LaTeX title block replaces it) and shifts headings up one level
--  * <!-- pdf:skip-start/end --> hides blocks; <!-- pdf:abstract-start/end --> boxes the abstract
--  * links between documents point to the PDFs on GitHub (relative .md links break inside a PDF)
--  * lone images become numbered figures; <img> tags become images
--  * tables get content-based column widths and styled headers

local REPO = "https://github.com/raisserv2/roomsense/blob/main/"

local function latex(s) return pandoc.RawInline("latex", s) end
local function latexb(s) return pandoc.RawBlock("latex", s) end

local function marker(b)
  if b.t == "RawBlock" and b.format == "html" then
    return b.text:match("<!%-%-%s*pdf:([%w%-]+)%s*%-%->")
  end
end

-- whole-document pass: markers and the title -------------------------------------------------
local function structure(doc)
  local out, skipping, title_done = {}, false, false
  for _, b in ipairs(doc.blocks) do
    local m = marker(b)
    if m == "skip-start" then skipping = true
    elseif m == "skip-end" then skipping = false
    elseif m == "abstract-start" then table.insert(out, latexb("\\begin{rsabstract}"))
    elseif m == "abstract-end" then table.insert(out, latexb("\\end{rsabstract}"))
    elseif skipping then -- drop
    elseif b.t == "Header" and b.level == 1 and not title_done then title_done = true
    elseif b.t == "HorizontalRule" then -- drop
    else table.insert(out, b) end
  end
  -- table of contents after the abstract (or at the top)
  if doc.meta.rstoc then
    local at = 1
    for i, b in ipairs(out) do
      if b.t == "RawBlock" and b.text == "\\end{rsabstract}" then at = i + 1 end
    end
    table.insert(out, at, latexb("{\\hypersetup{linkcolor=ink}\\setcounter{tocdepth}{1}\\tableofcontents}\\bigskip"))
  end
  doc.blocks = out
  return doc
end

-- strip a leading "Abstract." label inside the abstract box
local function strip_abstract_label(doc)
  for i, b in ipairs(doc.blocks) do
    if b.t == "RawBlock" and b.text == "\\begin{rsabstract}" then
      local p = doc.blocks[i + 1]
      if p and p.t == "Para" and p.content[1] and p.content[1].t == "Strong"
         and pandoc.utils.stringify(p.content[1]):match("^Abstract") then
        p.content:remove(1)
        if p.content[1] and p.content[1].t == "Space" then p.content:remove(1) end
      end
    end
  end
  return doc
end

-- headings ------------------------------------------------------------------------------------
local function Header(h)
  h.level = math.max(1, h.level - 1)
  local first = h.content[1]
  if first and first.t == "Str" then
    local num = first.text:match("^(%d+[%.%d]*)$")
    if num then
      num = num:gsub("%.$", "")
      h.content:remove(1)
      if h.content[1] and h.content[1].t == "Space" then h.content:remove(1) end
      h.content:insert(1, latex("\\secnum{" .. num .. "}"))
    end
  end
  h.classes:insert("unnumbered")
  return h
end

-- links ---------------------------------------------------------------------------------------
local function Link(l)
  local t = l.target
  if t:match("^%a[%w+.-]*:") or t:match("^#") then return nil end
  local file, anchor = t:match("^([^#]+)(#?.*)$")
  local doc = file and file:match("^(0%d%-[%w%-]+)%.md$")
  if doc then
    l.target = REPO .. "docs/pdf/" .. doc .. ".pdf"
  elseif file then
    l.target = REPO .. "docs/" .. file .. anchor
  end
  return l
end

-- inline code: colour, and allow line breaks after _ . / - :
local function Code(c)
  local out = pandoc.List({ latex("{\\inlinecode{}") })
  -- break at underscores only in long names, so at_desk stays whole
  local seps = #c.text > 16 and "_%./:%-|," or "%./:|,"
  for piece, sep in c.text:gmatch("([^" .. seps .. "]*)([" .. seps .. "]?)") do
    if piece ~= "" or sep ~= "" then
      out:insert(pandoc.Code(piece .. sep))
      if sep ~= "" then out:insert(latex("\\allowbreak{}")) end
    end
  end
  out:insert(latex("}"))
  return out
end

-- code blocks without a language still get the shaded box
local function CodeBlock(cb)
  if #cb.classes == 0 then cb.classes:insert("default") end
  return cb
end

-- images and figures --------------------------------------------------------------------------
local function html_img(text)
  local src = text:match('<img[^>]-src="([^"]+)"')
  if not src then return nil end
  local alt = text:match('alt="([^"]*)"') or ""
  local img = pandoc.Image({ pandoc.Str(alt) }, src)
  img.attributes.height = "7.5cm"
  return img, alt
end

local function figure(img, alt)
  return pandoc.Figure({ pandoc.Plain({ img }) }, { long = { pandoc.Plain(pandoc.Inlines(alt)) } })
end

local function Para(p)
  if #p.content == 1 and p.content[1].t == "Image" then
    local img = p.content[1]
    return figure(img, img.caption)
  end
  if #p.content == 1 and p.content[1].t == "RawInline" and p.content[1].format == "html" then
    local img, alt = html_img(p.content[1].text)
    if img then return figure(img, alt) end
  end
end

local function RawBlock(b)
  if b.format == "html" then
    local img, alt = html_img(b.text)
    if img then return figure(img, alt) end
    return {}
  end
end

-- tables --------------------------------------------------------------------------------------
local function cell_len(cell)
  return #pandoc.utils.stringify(cell.contents)
end

local function Table(t)
  local ncol = #t.colspecs
  local maxlen, maxword = {}, {}
  for j = 1, ncol do maxlen[j] = 3; maxword[j] = 3 end
  local function scan(rows)
    for _, r in ipairs(rows) do
      for j, c in ipairs(r.cells) do
        if j <= ncol then
          local text = pandoc.utils.stringify(c.contents)
          maxlen[j] = math.max(maxlen[j], #text)
          for word in text:gmatch("%S+") do maxword[j] = math.max(maxword[j], math.min(#word, 18)) end
        end
      end
    end
  end
  scan(t.head.rows)
  for _, body in ipairs(t.bodies) do scan(body.body) end

  -- drop a header row whose cells are all empty
  local empty_head = true
  for _, r in ipairs(t.head.rows) do
    for _, c in ipairs(r.cells) do
      if pandoc.utils.stringify(c.contents) ~= "" then empty_head = false end
    end
  end
  if empty_head then t.head.rows = {} end

  -- style header cells
  for _, r in ipairs(t.head.rows) do
    for _, c in ipairs(r.cells) do
      for _, blk in ipairs(c.contents) do
        if blk.t == "Plain" or blk.t == "Para" then
          blk.content:insert(1, latex("{\\rsthead{}"))
          blk.content:insert(latex("}"))
        end
      end
    end
  end

  -- widths from content when the table would not fit on one line:
  -- proportional to (damped) content length, but never narrower than the longest word
  local total, w, minw = 0, {}, {}
  for j = 1, ncol do
    w[j] = math.min(maxlen[j], 70) ^ 0.8
    minw[j] = (maxword[j] + 1.5) / 92
    total = total + maxlen[j]
  end
  if total > 80 then
    local s = 0
    for j = 1, ncol do s = s + w[j] end
    local f, fixed = {}, 0
    for j = 1, ncol do f[j] = math.max(w[j] / s, minw[j]) ; fixed = fixed + f[j] end
    for j = 1, ncol do t.colspecs[j][2] = f[j] / fixed * 0.98 end
  end
  return t
end

return {
  { Pandoc = structure },
  { Pandoc = strip_abstract_label },
  { Header = Header, Link = Link, Code = Code, CodeBlock = CodeBlock, Para = Para, RawBlock = RawBlock, Table = Table },
}
