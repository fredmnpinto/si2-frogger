{
  description = "Frogger Agent - Python project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python311;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "frogger-agent";
        KERAS_BACKEND = "jax";

        buildInputs = [
          python
          pkgs.stdenv.cc.cc.lib
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          if [ ! -d ".venv" ]; then
            echo "Creating Python virtual environment..."
            ${python}/bin/python -m venv .venv
          fi

          source .venv/bin/activate

          if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt >/dev/null 2>&1 || true
          fi
        '';
      };
    };
}
