import slam


def main() -> None:
    print("slam import ok")
    print("sensors:", [name for name in dir(slam.Sensor) if name.isupper()])


if __name__ == "__main__":
    main()
