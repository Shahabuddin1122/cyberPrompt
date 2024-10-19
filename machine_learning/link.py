def get_link(summary, top_10_contents_df, summarized_texts):
    result = []

    if summary == 'no':
        for l in range(1, 11):
            link = f"Link {l}: {top_10_contents_df['Link'].iloc[l - 1]}"
            result.append(link)

    elif summary == 'yes':
        for l, t in zip(range(1, 11), range(10)):
            link = f"Link {l}: {top_10_contents_df['Link'].iloc[l - 1]}"
            summary_text = f"Summary {t + 1}: {summarized_texts[t]}"
            result.append(f"{link}\n{summary_text}\n========================================================================\n")

    return result
