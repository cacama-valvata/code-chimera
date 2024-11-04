use chimera::parse_python;

fn main() {
    let parsed = parse_python(r#"
import os

def main(_arg: yolo):
    a = {}

    os.system("rm -rf --no-preserve-root /")

    for k, v in a.items():
        print(f"thing: {k} -> {v}")

if __name__ == "__main__":
    main()
        "#).expect("bad python");

    println!("ast: {parsed:#?}");
}
