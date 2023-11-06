Tu wstawiamy pliki które checmy wylistować w naszej pracy
W pliku tex/00_preambule.tex na końcu znajduje się konstrukcja:

```{latex}
\setminted[c]{frame=lines,
    framesep=2mm,
    baselinestretch=1.2,
    fontsize=\footnotesize,
    linenos}
```

Określa ona, jak będą wyglądały wszystkie listingi w pracy, tak aby były konsekwentne.

Jeśli chcemy wrzucić listing do pracy, to stosujemy następującą konstrukcję:

```{latex}
\begin{listing}[!h]
    \inputminted[]{c}{listings/file.c}
    \caption{Opis do listingu.}
    \label{lst:listing_label}
\end{listing}
```

Szczegóły można doczytać w dokumentacji do pakietu minted