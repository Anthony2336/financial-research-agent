"""Lookup and intent resolution for static research recipes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from fra.domain import Intent
from fra.skills.models import ResearchRecipe, SkillName


@dataclass(frozen=True, slots=True)
class ExactIntentRecipeContract:
    """Versioned exact ordered recipe composition for dispatchable research intents."""

    version: str
    ordered_recipe_names: Mapping[Intent, tuple[SkillName, ...]]

    def __post_init__(self) -> None:
        normalized: dict[Intent, tuple[SkillName, ...]] = {}
        for intent, recipe_names in self.ordered_recipe_names.items():
            if not recipe_names:
                raise ValueError(f"dispatch contract {self.version} has no recipes for {intent}")
            normalized[intent] = tuple(recipe_names)
        object.__setattr__(
            self,
            "ordered_recipe_names",
            MappingProxyType(normalized),
        )


class ResearchRecipeRegistry:
    """An ordered collection of immutable, uniquely identified recipes."""

    def __init__(
        self,
        recipes: Sequence[ResearchRecipe],
        *,
        dispatch_contract: ExactIntentRecipeContract | None = None,
    ) -> None:
        identities = tuple((recipe.name, recipe.version) for recipe in recipes)
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate recipe name and version")
        self._recipes = tuple(recipes)
        self._recipes_by_name = {recipe.name: recipe for recipe in self._recipes}
        if dispatch_contract is not None:
            self._validate_dispatch_contract(dispatch_contract)

    def get(self, name: SkillName) -> ResearchRecipe:
        """Return the registered recipe for a supported skill name."""
        return self._recipes_by_name[name]

    def resolve(self, intent: Intent) -> tuple[ResearchRecipe, ...]:
        """Return applicable recipes in explicit declaration order."""
        return tuple(recipe for recipe in self._recipes if intent in recipe.accepted_intents)

    def _validate_dispatch_contract(self, contract: ExactIntentRecipeContract) -> None:
        contract_intents = frozenset(contract.ordered_recipe_names)
        for recipe in self._recipes:
            if recipe.input_schema != "ResearchInput":
                continue
            unexpected_intents = set(recipe.accepted_intents) - contract_intents
            if unexpected_intents:
                rendered = ", ".join(sorted(intent.value for intent in unexpected_intents))
                raise ValueError(
                    f"dispatch contract {contract.version} has no declaration for {rendered}"
                )
        for intent, expected_names in contract.ordered_recipe_names.items():
            actual_names = tuple(recipe.name for recipe in self.resolve(intent))
            if actual_names != expected_names:
                raise ValueError(
                    f"dispatch contract {contract.version} mismatch for {intent.value}: "
                    f"expected {[name.value for name in expected_names]} "
                    f"got {[name.value for name in actual_names]}"
                )
