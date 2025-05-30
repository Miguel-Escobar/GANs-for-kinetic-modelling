{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
    nixpkgs-python.inputs = { nixpkgs.follows = "nixpkgs"; };

    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
    devenv.inputs.nixpkgs.follows = "nixpkgs";
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
    rocmSupport = true;
  };

  outputs = { self, nixpkgs, devenv, systems, ... } @ inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      packages = forEachSystem (system: {
        devenv-up = self.devShells.${system}.default.config.procfileScript;
        devenv-test = self.devShells.${system}.default.config.test;
      });

      devShells = forEachSystem
        (system:
          let
            pkgs = import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true;
                rocmSupport = true;
              };
            };
          in
          {
            default = devenv.lib.mkShell {
              inherit inputs pkgs;
              modules = [
                ({ pkgs, config, inputs, ... }:
                  {
                    # https://devenv.sh/packages/
                    packages = [
                      pkgs.git
                      pkgs.python312Packages.torchWithRocm
                      pkgs.python312Packages.torchvision
                    ];

                    # https://devenv.sh/languages/
                    languages.python.enable = true;
                    languages.python.version = "3.12.8";
                    languages.python.venv.enable = true;
                    languages.python.venv.requirements = ./requirements.txt;

                    scripts.hello.exec = ''
                      echo hello!
                    '';

                    # https://devenv.sh/tests/
                    enterTest = ''
                      echo "Running tests"
                      git --version | grep --color=auto "${pkgs.git.version}"
                    '';
                  })
              ];
            };
          });
    };
}
