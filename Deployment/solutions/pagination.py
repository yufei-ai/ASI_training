def get_companies(self):
    page = 0

    companies = self._get_companies(page)
    items = companies['items']
    maximum_items_per_page = companies['maximum_items_per_page']

    all_items = items

    while len(items) == maximum_items_per_page:
        page += 1
        items = self._get_companies(page)['items']
        all_items.extend(items)

    return all_items
