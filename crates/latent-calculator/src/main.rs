//! LatCal terminal — read a natural-language command, print the answer.

use std::io::{self, BufRead, Write};

use latent_calculator::Calculator;

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            break;
        }
        match Calculator::parse(trimmed) {
            Ok(answer) => {
                let _ = writeln!(out, "{}", answer.to_sentence());
            }
            Err(_) => {
                let _ = writeln!(out, "sorry, I could not understand that");
            }
        }
        let _ = out.flush();
    }
}
