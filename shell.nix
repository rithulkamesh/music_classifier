{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    ffmpeg
    # Add libsndfile and other audio dependencies
    libsndfile
    libjack2
    
    # Python dependencies will be managed by uv
    python311Packages.pip
  ];
  
  shellHook = ''
    echo "NixOS environment for Music Classifier"
    echo "Running setup script..."
    ./setup.sh
  '';
}
