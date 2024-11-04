use rustpython_parser::{Parse, ast, ParseError};

// this is sorta useless rn but will hopefully do more in the future :^)
pub fn parse_python(code: &str) -> Result<ast::Suite, ParseError> {
    ast::Suite::parse(code, "<embedded>")
}

// stuff to be agnostic of (maybe):
// - type annotations
// - identifiers?
